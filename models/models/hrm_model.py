"""
HRM (Hierarchical Reasoning Model): A brain-inspired recurrent architecture for complex reasoning.

This model implements the architecture described in "Hierarchical Reasoning Model" paper, featuring:
- Two-level hierarchical processing (High-level and Low-level modules)
- Multi-timescale dynamics inspired by neural oscillations
- Hierarchical convergence mechanism
- 1-step gradient approximation for efficient training
- Adaptive Computation Time (ACT) with Q-learning
- Deep supervision for stable training

The model achieves strong performance on complex reasoning tasks like ARC-AGI, Sudoku, and Maze navigation
with only ~27M parameters and minimal training data.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

# Import components from blocks
from models.blocks.normalization import RMSNorm, DynamicTanh
from models.blocks.positional_encoding import RoPE
from models.blocks.mlp import MLP

# Import utility functions
from train.train_utils import estimate_mfu as utils_estimate_mfu

@dataclass
class HRMConfig:
    """Configuration for HRM."""
    # Architecture
    n_layer: int = 1  # HRM uses a single layer with recurrence, not stacked layers
    n_embd: int = 512  # Model dimension
    n_head: int = 8  # Number of attention heads
    n_inner: Optional[int] = None  # Inner dimension for MLP. If None, will be 4*n_embd
    vocab_size: int = 50304
    block_size: int = 900  # Maximum sequence length (30x30 for ARC-AGI)
    
    # HRM specific parameters
    cycles_per_segment: int = 2  # N: Number of high-level cycles per segment
    steps_per_cycle: int = 3  # T: Number of low-level steps per cycle
    max_segments: int = 8  # M_max: Maximum number of segments for ACT
    min_segments: int = 1  # Minimum segments before allowing halt
    gradient_steps: int = -1  # Number of steps with gradients (-1 for all, 1 for 1-step approx)
    
    # ACT (Adaptive Computation Time) parameters
    use_act: bool = True  # Enable adaptive computation
    act_epsilon: float = 0.1  # Epsilon for Q-learning exploration
    ponder_loss_weight: float = 0.01  # Weight for ponder loss
    halt_bias_init: float = -2.0  # Initial bias for halt head
    
    # Dropout and regularization
    dropout: float = 0.1
    bias: bool = False
    
    # Training
    use_gradient_checkpointing: bool = False  # HRM uses 1-step gradient instead
    use_deep_supervision: bool = True  # Enable deep supervision
    n_supervision_segments: int = 4  # Number of segments for deep supervision
    deq_one_step: bool = False  # Use 1-step gradient approximation (for memory efficiency)
    init_device: str = 'cuda'
    label_smoothing: float = 0.0
    
    # Learning rate and optimization
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0

    # DynamicTanh
    dyt_init_alpha: float = 0.5
    
    def __post_init__(self):
        # Set inner dimension if not provided
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd


class HRMBlock(nn.Module):
    """
    Transformer block with Post-Norm architecture.
    Used as the building block for both H and L modules.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Post-Norm architecture: normalization after residual
        self.norm1 = RMSNorm(config.n_embd)
        self.norm2 = RMSNorm(config.n_embd)
        
        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            config.n_embd,
            config.n_head,
            dropout=config.dropout,
            batch_first=True,
            bias=config.bias
        )
        
        # Feed-forward network with SwiGLU activation
        self.mlp = SwiGLUFFN(config.n_embd, config.n_inner, config.dropout)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Attention with Post-Norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout(attn_out)
        
        # FFN with Post-Norm
        x = x + self.dropout(self.mlp(self.norm2(x)))
        
        return x


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    Combines SiLU (Swish) activation with gating mechanism.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # Gate
        self.w2 = nn.Linear(d_model, d_ff, bias=False)  # Value
        self.w3 = nn.Linear(d_ff, d_model, bias=False)  # Output
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # SwiGLU: gate * swish(value)
        gate = self.w1(x)
        value = self.w2(x)
        hidden = F.silu(gate) * value  # SiLU = x * sigmoid(x)
        output = self.w3(hidden)
        return self.dropout(output)


class LowLevelModule(nn.Module):
    """
    Low-level module for fast, detailed computations.
    
    Inspired by lower cortical areas that process immediate sensory/motor information.
    Updates at every timestep (fast timescale, like gamma waves).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer_block = HRMBlock(config)
    
    def forward(self, z_L, z_H, x_tilde, attn_mask=None, key_padding_mask=None):
        """
        Update equation: z_L^i = f_L(z_L^{i-1}, z_H^{i-1}, x̃)
        
        Args:
            z_L: Previous low-level state
            z_H: Current high-level state (fixed during cycle)
            x_tilde: Encoded input
            attn_mask: Causal attention mask
            key_padding_mask: Padding mask
        """
        # Simple fusion of states and input
        z_input = z_L + z_H + x_tilde
        return self.transformer_block(z_input, attn_mask, key_padding_mask)


class HighLevelModule(nn.Module):
    """
    High-level module for abstract planning and strategy.
    
    Inspired by higher cortical areas like prefrontal cortex.
    Updates only every T timesteps (slow timescale, like theta waves).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer_block = HRMBlock(config)
    
    def forward(self, z_H, z_L, attn_mask=None, key_padding_mask=None):
        """
        Update equation: z_H^i = f_H(z_H^{i-1}, z_L^{i-1})
        Only updates when i ≡ 0 (mod T)
        
        Args:
            z_H: Previous high-level state
            z_L: Final low-level state from completed cycle
            attn_mask: Causal attention mask
            key_padding_mask: Padding mask
        """
        z_input = z_H + z_L
        return self.transformer_block(z_input, attn_mask, key_padding_mask)


class HRMInner(nn.Module):
    """
    Inner HRM module implementing hierarchical convergence.
    
    The key mechanism that allows deep computation:
    - L-module converges locally within each cycle
    - H-module updates establish new context
    - This prevents premature convergence of standard RNNs
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.H_module = HighLevelModule(config)
        self.L_module = LowLevelModule(config)
    
    def forward(self, z_H, z_L, x_tilde, attn_mask=None, key_padding_mask=None):
        """
        Execute one step of hierarchical computation.
        
        Note: The actual hierarchical timing (when H updates) is handled
        by the main HRM model's forward_segment method.
        """
        # L-module always updates
        z_L_new = self.L_module(z_L, z_H, x_tilde, attn_mask, key_padding_mask)
        
        # H-module update (called conditionally by parent)
        z_H_new = self.H_module(z_H, z_L_new, attn_mask, key_padding_mask)
        
        return z_H_new, z_L_new


class ACTController(nn.Module):
    """
    Adaptive Computation Time controller using Q-learning.
    
    Inspired by the brain's ability to dynamically allocate cognitive resources.
    Implements "thinking fast and slow" by learning when to halt computation.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_segments = config.max_segments
        self.min_segments = config.min_segments
        self.epsilon = config.act_epsilon
        
        # Q-head: predicts Q(halt) and Q(continue)
        self.q_head = nn.Sequential(
            nn.Linear(config.n_embd, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [Q_halt, Q_continue]
            nn.Sigmoid()
        )
    
    def forward(self, z_H):
        """Compute Q-values for halt/continue decision."""
        # Pool over sequence dimension
        z_H_pooled = z_H.mean(dim=1)  # [batch_size, d_model]
        q_values = self.q_head(z_H_pooled)
        return q_values
    
    def should_halt(self, z_H, segment_idx):
        """
        Decide whether to halt computation.
        
        Args:
            z_H: Current high-level state
            segment_idx: Current segment index
        
        Returns:
            Boolean tensor indicating halt decision for each batch element
        """
        batch_size = z_H.size(0)
        device = z_H.device
        
        # Create conditions as tensors for torch.compile compatibility
        at_max = segment_idx >= self.max_segments - 1
        before_min = segment_idx < self.min_segments
        
        # Compute Q-values
        q_values = self.forward(z_H)
        q_halt = q_values[:, 0]
        q_continue = q_values[:, 1]
        
        # Epsilon-greedy decision
        random_halt = torch.rand(batch_size, device=device) < 0.5
        exploit_halt = q_halt > q_continue
        
        # Mix exploration and exploitation
        use_random = torch.rand(batch_size, device=device) < self.epsilon
        halt_decision = torch.where(use_random, random_halt, exploit_halt)
        
        # Apply constraints using torch.where
        # Must halt at maximum
        halt_decision = torch.where(
            torch.tensor(at_max, device=device).expand(batch_size),
            torch.ones(batch_size, dtype=torch.bool, device=device),
            halt_decision
        )
        
        # Cannot halt before minimum
        halt_decision = torch.where(
            torch.tensor(before_min, device=device).expand(batch_size),
            torch.zeros(batch_size, dtype=torch.bool, device=device),
            halt_decision
        )
        
        return halt_decision


class HRM(nn.Module):
    """
    Hierarchical Reasoning Model.
    
    A brain-inspired architecture that achieves deep computation through:
    1. Hierarchical processing with temporal separation
    2. Convergence hierarchy preventing early saturation
    3. 1-step gradient approximation for efficient training
    4. Adaptive computation time for resource allocation
    5. Deep supervision for stable learning
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embeddings = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        # Hierarchical modules
        self.inner_model = HRMInner(config)
        
        # Output heads
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # ACT components
        if config.use_act:
            self.act_controller = ACTController(config)
            self.halt_head = nn.Sequential(
                nn.Linear(config.n_embd, 1),
                nn.Sigmoid()
            )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Set halt bias for ACT
        if config.use_act:
            with torch.no_grad():
                self.halt_head[0].bias.fill_(config.halt_bias_init)
        
        # Register position indices buffer
        self.register_buffer("pos_ids", torch.arange(config.block_size).unsqueeze(0))
        
        # Parameter count
        self.param_count = sum(p.numel() for p in self.parameters())
        print(f"HRM initialized with {self.param_count/1e6:.2f}M parameters")
    
    def _init_weights(self, module):
        """Initialize weights with scaled initialization."""
        if isinstance(module, nn.Linear):
            # Use scaled initialization based on fan-in
            std = 0.02  # Default std
            if hasattr(module, 'weight'):
                # Xavier/He initialization
                fan_in = module.weight.size(1)
                std = math.sqrt(2.0 / fan_in) * 0.5  # Scale down a bit for stability
            nn.init.normal_(module.weight, mean=0.0, std=std)
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        elif isinstance(module, (RMSNorm, DynamicTanh)):
            if hasattr(module, 'weight'):
                nn.init.ones_(module.weight)
    
    def compute_embeddings(self, input_ids):
        """Combine token and position embeddings."""
        batch_size, seq_len = input_ids.shape
        token_emb = self.token_embeddings(input_ids)
        pos_emb = self.pos_embeddings(self.pos_ids[:, :seq_len])
        return self.dropout(token_emb + pos_emb)
    
    def forward_segment(self, z_H, z_L, x_embedded, attention_mask=None):
        """
        Execute one segment (N cycles × T steps).
        
        This implements the hierarchical convergence mechanism where:
        - L-module converges over T steps
        - H-module updates once per cycle
        - Total computation: N × T timesteps
        """
        batch_size, seq_len = x_embedded.shape[:2]
        device = x_embedded.device
        
        # Prepare attention masks
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        
        # Track convergence metrics
        convergence_metrics = {
            'l_residuals': [],
            'h_residuals': []
        }
        
        # Execute N cycles
        for cycle in range(self.config.cycles_per_segment):
            # T steps of L-module (fast timescale)
            for step in range(self.config.steps_per_cycle):
                z_L_prev = z_L.clone()
                
                # L-module update
                z_L = self.inner_model.L_module(
                    z_L, z_H, x_embedded,
                    attn_mask=causal_mask,
                    key_padding_mask=key_padding_mask
                )
                
                # Track convergence
                l_residual = (z_L - z_L_prev).norm(dim=-1).mean()
                convergence_metrics['l_residuals'].append(l_residual)
                
                # H-module update at end of cycle (slow timescale)
                if step == self.config.steps_per_cycle - 1:
                    z_H_prev = z_H.clone()
                    z_H = self.inner_model.H_module(
                        z_H, z_L,
                        attn_mask=causal_mask,
                        key_padding_mask=key_padding_mask
                    )
                    h_residual = (z_H - z_H_prev).norm(dim=-1).mean()
                    convergence_metrics['h_residuals'].append(h_residual)
        
        return z_H, z_L, convergence_metrics
    
    def compute_halting_probability(self, z_H, step_idx):
        """
        Compute probability of halting computation.
        
        Uses the halt head to predict halting probability based on
        the current high-level state.
        """
        if not self.config.use_act:
            # Without ACT, always halt after one segment
            return torch.ones(z_H.size(0), z_H.size(1), device=z_H.device)
        
        p_halt = self.halt_head(z_H).squeeze(-1)
        
        # Clamp for numerical stability
        eps = 1e-6
        p_halt = p_halt.clamp(eps, 1 - eps)
        
        # Force halting at maximum using torch.where for compile compatibility
        at_max = step_idx >= self.config.max_segments - 1
        p_halt = torch.where(
            torch.tensor(at_max, device=z_H.device),
            torch.ones_like(p_halt),
            p_halt
        )
        
        return p_halt
    
    def forward_with_1step_gradient(self, z, x_embedded, attention_mask=None):
        """
        Forward pass with configurable gradient steps.
        
        This implements flexible gradient computation:
        - gradient_steps = 1: Only final step (memory efficient)
        - gradient_steps = -1: All steps (better learning)
        - gradient_steps = N: Last N steps
        """
        z_H, z_L = z
        
        # Prepare attention masks
        batch_size, seq_len = x_embedded.shape[:2]
        device = x_embedded.device
        
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        
        total_steps = self.config.cycles_per_segment * self.config.steps_per_cycle
        
        # Determine how many steps to compute with gradients
        if self.config.gradient_steps == -1:
            # All steps with gradients (best learning, more memory)
            grad_steps = total_steps
        else:
            # Last N steps with gradients
            grad_steps = min(self.config.gradient_steps, total_steps)
        
        no_grad_steps = total_steps - grad_steps
        
        # Forward without gradient for initial steps
        if no_grad_steps > 0:
            with torch.no_grad():
                for i in range(no_grad_steps):
                    # L-module update
                    z_L = self.inner_model.L_module(z_L, z_H, x_embedded, causal_mask, key_padding_mask)
                    
                    # H-module update at cycle boundaries
                    if (i + 1) % self.config.steps_per_cycle == 0:
                        z_H = self.inner_model.H_module(z_H, z_L, causal_mask, key_padding_mask)
        
        # Forward with gradient for remaining steps
        for i in range(no_grad_steps, total_steps):
            # L-module update
            z_L = self.inner_model.L_module(z_L, z_H, x_embedded, causal_mask, key_padding_mask)
            
            # H-module update at cycle boundaries
            if (i + 1) % self.config.steps_per_cycle == 0:
                z_H = self.inner_model.H_module(z_H, z_L, causal_mask, key_padding_mask)
        
        return (z_H, z_L)
    
    def forward(self, input_ids, labels=None, attention_mask=None, return_intermediates=False):
        """
        Forward pass with optional ACT and deep supervision.
        
        Returns:
            dict with keys:
            - loss: Total loss (if labels provided)
            - logits: Output predictions
            - ponder_cost: Average computation used
            - segments_used: Number of segments executed
            - convergence_data: Convergence metrics (if requested)
        """
        # Compute embeddings
        x_embedded = self.compute_embeddings(input_ids)
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize hidden states
        z_L = torch.zeros_like(x_embedded)
        z_H = torch.zeros_like(x_embedded)
        
        # ACT: Adaptive computation
        halting_probs = []
        remainders = torch.ones(batch_size, seq_len, device=device)
        accumulated_z_H = torch.zeros_like(z_H)
        n_updates = torch.zeros(batch_size, seq_len, device=device)
        
        segments_outputs = []
        convergence_data = []
        
        # Execute segments with optional early stopping
        for segment_idx in range(self.config.max_segments):
            # Forward one segment
            if self.training and self.config.deq_one_step:
                # Use configurable gradient steps for training
                z_H, z_L = self.forward_with_1step_gradient((z_H, z_L), x_embedded, attention_mask)
                conv_metrics = None
            else:
                # Full forward for normal training or inference
                z_H, z_L, conv_metrics = self.forward_segment(
                    z_H, z_L, x_embedded, attention_mask
                )
            
            if return_intermediates and conv_metrics:
                convergence_data.append(conv_metrics)
            
            # Compute halting probability
            p_halt = self.compute_halting_probability(z_H, segment_idx)
            is_last = (segment_idx == self.config.max_segments - 1)
            
            # Weighted contribution (ACT mechanism)
            # Use torch.where for compile compatibility
            contrib = torch.where(
                is_last.unsqueeze(-1) if torch.is_tensor(is_last) else torch.tensor(is_last, device=device).unsqueeze(-1),
                remainders,
                remainders * p_halt
            )
            
            halting_probs.append(contrib)
            accumulated_z_H += contrib.unsqueeze(-1) * z_H
            
            # Update remainders
            # Use torch operations for compile compatibility
            remainders = torch.where(
                is_last.unsqueeze(-1) if torch.is_tensor(is_last) else torch.tensor(is_last, device=device).unsqueeze(-1),
                remainders,  # Keep same if last
                remainders * (1 - p_halt)
            )
            # Track ponder cost
            n_updates = n_updates + torch.where(
                is_last.unsqueeze(-1) if torch.is_tensor(is_last) else torch.tensor(is_last, device=device).unsqueeze(-1),
                torch.zeros_like(remainders),
                remainders
            )
            
            # Generate output for this segment
            segment_logits = self.lm_head(z_H)
            segments_outputs.append(segment_logits)
            
            # Early stopping if all samples have halted
            # Note: Removed early stopping to be compatible with torch.compile
            # The loop will continue until max_segments even if all samples have halted
            # This is a small efficiency trade-off for compilation compatibility
        
        # Final output using weighted combination
        if self.config.use_act:
            # Apply layer norm before final projection for stability with large vocab
            if self.config.vocab_size > 10000:
                accumulated_z_H = F.layer_norm(accumulated_z_H, [accumulated_z_H.size(-1)])
            final_logits = self.lm_head(accumulated_z_H)
        else:
            # For non-ACT, use the last segment output
            if self.config.vocab_size > 10000:
                # Apply layer norm before projection for stability
                z_H_final = F.layer_norm(z_H, [z_H.size(-1)])
                final_logits = self.lm_head(z_H_final)
            else:
                final_logits = segments_outputs[-1]
        
        # Compute losses
        loss = None
        lm_loss = None
        ponder_loss = None
        
        if labels is not None:
            # Language modeling loss
            shift_logits = final_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing
            )
            lm_loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            
            # Ponder loss (regularizes computation time)
            if self.config.use_act:
                ponder_loss = torch.mean(n_updates) * self.config.ponder_loss_weight
                loss = lm_loss + ponder_loss
            else:
                loss = lm_loss
        
        # Prepare output
        output = {
            "loss": loss,
            "logits": final_logits,
            "lm_loss": lm_loss,
            "ponder_loss": ponder_loss,
            "halting_probs": torch.stack(halting_probs) if halting_probs else None,
            "ponder_cost": torch.mean(n_updates) if self.config.use_act else None,
            "segments_used": segment_idx + 1
        }
        
        if return_intermediates:
            output["segments_outputs"] = segments_outputs
            output["convergence_data"] = convergence_data
        
        return output
    
    def generate(self, idx=None, max_new_tokens=20, temperature=1.0, top_k=None, prompt=None, gen_length=None):
        """
        Generate text autoregressively.
        
        Uses full computation (not 1-step) for best quality generation.
        """
        # Handle compatibility with train_utils.generate_text
        if prompt is not None:
            idx = prompt
        elif idx is None:
            raise ValueError("Either idx or prompt must be provided")
        
        # Determine how many tokens to generate
        if gen_length is not None:
            max_new_tokens = gen_length
        self.eval()
        
        for _ in range(max_new_tokens):
            # Ensure sequence doesn't exceed maximum length
            # Use slicing that works with compile
            seq_len = idx.size(1)
            if seq_len > self.config.block_size:
                idx_cond = idx[:, -self.config.block_size:]
            else:
                idx_cond = idx
            
            # Forward pass
            with torch.no_grad():
                outputs = self(idx_cond)
                logits = outputs["logits"]
            
            # Extract logits for last position
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k sampling if needed
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        # Return tuple to match expected signature
        return idx, None
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, optimizer_type=None, **kwargs):
        """
        Configure AdamW optimizer with weight decay.
        
        Separates parameters into decay and no-decay groups.
        """
        # Separate parameters
        decay = set()
        no_decay = set()
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, (nn.Linear, nn.Embedding)):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, RMSNorm):
                    no_decay.add(fpn)
        
        # Create parameter groups
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        
        # Use fused AdamW if available
        use_fused = device_type == 'cuda'
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            fused=use_fused
        )
        
        return optimizer


def train_with_deep_supervision(model, dataloader, optimizer, config):
    """
    Training with deep supervision.
    
    Implements the training procedure from the paper:
    - Multiple segments per batch
    - 1-step gradient approximation
    - Detached states between segments
    """
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        input_ids, labels = batch
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize states
        z_H = torch.zeros(batch_size, seq_len, config.n_embd, device=device)
        z_L = torch.zeros(batch_size, seq_len, config.n_embd, device=device)
        
        # Deep supervision: multiple segments
        for segment in range(config.n_supervision_segments):
            # Forward with 1-step gradient
            x_embedded = model.compute_embeddings(input_ids)
            (z_H_new, z_L_new) = model.forward_with_1step_gradient((z_H, z_L), x_embedded)
            
            # Compute loss
            logits = model.lm_head(z_H_new)
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                labels.view(-1)
            )
            
            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            
            # Detach states for next segment (crucial!)
            z_H = z_H_new.detach()
            z_L = z_L_new.detach()
            
            total_loss += loss.item()
    
    return total_loss / (len(dataloader) * config.n_supervision_segments)


def create_hrm_model(
    size: str = 'small',
    n_layer: Optional[int] = None,
    n_embd: Optional[int] = None,
    n_head: Optional[int] = None,
    vocab_size: int = 50304,
    block_size: int = 900,
    dropout: float = 0.1,
    **kwargs
):
    """
    Create an HRM model with predefined sizes.
    
    Args:
        size: Model size ('small', 'medium', 'large')
        n_layer: Override number of layers (HRM uses 1 with recurrence)
        n_embd: Override embedding dimension
        n_head: Override number of heads
        vocab_size: Vocabulary size
        block_size: Maximum sequence length
        dropout: Dropout probability
        **kwargs: Additional configuration arguments
    
    Returns:
        HRM: Configured model
    """
    # Define model sizes (matching ~27M params for small)
    sizes = {
        'small': {
            'n_layer': 1,  # HRM uses recurrence, not stacking
            'n_embd': 512,
            'n_head': 8,
            'cycles_per_segment': 2,
            'steps_per_cycle': 3,
        },
        'medium': {
            'n_layer': 1,
            'n_embd': 768,
            'n_head': 12,
            'cycles_per_segment': 3,
            'steps_per_cycle': 4,
        },
        'large': {
            'n_layer': 1,
            'n_embd': 1024,
            'n_head': 16,
            'cycles_per_segment': 4,
            'steps_per_cycle': 4,
        }
    }
    
    if size not in sizes:
        raise ValueError(f"Unknown model size: {size}, valid sizes are {list(sizes.keys())}")
    
    config_dict = sizes[size].copy()
    
    # Override with explicit parameters
    if n_layer is not None:
        config_dict['n_layer'] = n_layer
    if n_embd is not None:
        config_dict['n_embd'] = n_embd
    if n_head is not None:
        config_dict['n_head'] = n_head
    
    # Add standard parameters
    config_dict.update({
        'vocab_size': vocab_size,
        'block_size': block_size,
        'dropout': dropout,
    })
    
    # Add any additional parameters
    config_dict.update(kwargs)
    
    # Create config and model
    config = HRMConfig(**config_dict)
    model = HRM(config)
    
    return model