"""
MOE-MLA Model: Mixture of Experts model using Multi-head Latent Attention.

This model combines:
- Multi-head Latent Attention (MLA) for efficient attention computation
- Mixture of Experts (MoE) with shared weights (75% shared, 25% specialized by default)
- FP8 training support for memory efficiency
- Dynamic Tanh (DyT) normalization
- GaLore2 optimizer support
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional, Dict, List, Tuple, Union, Any
from dataclasses import dataclass

# Import MLA components
from models.blocks.mla import MLA
from models.blocks.mla_fp8 import MLA_FP8
from models.blocks.normalization import RMSNorm, DynamicTanh
from models.blocks.positional_encoding import RoPE
from models.blocks.tensor_utils import isolate_tensor

# Import utility functions
from train.train_utils import estimate_mfu as utils_estimate_mfu

@dataclass
class MOEMLAConfig:
    """Configuration for MOE-MLA Model."""
    # Architecture
    n_layer: int = 24
    n_embd: int = 2048
    n_head: int = 16
    n_inner: Optional[int] = None  # Inner dimension for MLP. If None, will be 4*n_embd
    vocab_size: int = 50304
    block_size: int = 4096
    
    # MLA specific parameters
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    
    # MoE parameters
    num_experts: int = 8
    experts_per_token: int = 2  # Top-k routing
    shared_weight_ratio: float = 0.75  # 75% of weights are shared between experts
    
    # Attention backend
    attention_backend: Optional[str] = None
    attn_impl: str = "absorb"  # "absorb" (optimized) or "naive" (standard)
    
    # RoPE
    original_max_seq_len: int = 4096
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: float = 10000.0
    
    # Precision
    use_fp8: bool = False  # Master switch for FP8 usage
    fp8_tile_size: int = 128  # Tile size for FP8 quantization
    
    # Dropout and regularization
    dropout: float = 0.0
    bias: bool = False
    
    # Training
    use_gradient_checkpointing: bool = True
    init_device: str = 'cuda'
    label_smoothing: float = 0.0
    
    # Dynamic Tanh (DyT) options
    use_dyt: bool = False
    dyt_alpha_init: float = 0.5
    
    def __post_init__(self):
        # Set inner dimension if not provided
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd


class SharedExpertMLP(nn.Module):
    """
    MLP with shared and specialized weights for experts.
    """
    def __init__(self, config, num_experts: int):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.shared_ratio = config.shared_weight_ratio
        
        # Calculate dimensions
        self.hidden_dim = config.n_inner
        shared_dim = int(self.hidden_dim * self.shared_ratio)
        specialized_dim = self.hidden_dim - shared_dim
        
        # Shared components (used by all experts)
        self.shared_up_proj = nn.Linear(config.n_embd, shared_dim, bias=False)
        self.shared_down_proj = nn.Linear(shared_dim, config.n_embd, bias=False)
        
        # Expert-specific components
        self.expert_up_proj = nn.ModuleList([
            nn.Linear(config.n_embd, specialized_dim, bias=False)
            for _ in range(num_experts)
        ])
        self.expert_down_proj = nn.ModuleList([
            nn.Linear(specialized_dim, config.n_embd, bias=False)
            for _ in range(num_experts)
        ])
        
        # Activation
        self.activation = nn.SiLU()
        
        # Initialize weights
        self._init_weights()
    
    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor, 
                expert_weights: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with expert routing.
        
        Args:
            x: Input tensor [batch_size, seq_len, n_embd]
            expert_indices: Selected expert indices [batch_size, seq_len, experts_per_token]
            expert_weights: Weights for selected experts [batch_size, seq_len, experts_per_token]
        """
        batch_size, seq_len, n_embd = x.shape
        experts_per_token = expert_indices.shape[-1]
        
        # Shared forward pass (all tokens)
        shared_hidden = self.activation(self.shared_up_proj(x))
        shared_output = self.shared_down_proj(shared_hidden)
        
        # Flatten batch and sequence dimensions
        x_flat = x.view(-1, n_embd)  # [batch_size * seq_len, n_embd]
        expert_indices_flat = expert_indices.view(-1, experts_per_token)  # [batch_size * seq_len, experts_per_token]
        expert_weights_flat = expert_weights.view(-1, experts_per_token)  # [batch_size * seq_len, experts_per_token]
        
        # Create one-hot encoding for expert assignments
        # [batch_size * seq_len, experts_per_token, num_experts]
        expert_one_hot = F.one_hot(expert_indices_flat, num_classes=self.num_experts).float()
        
        # Compute all expert outputs in parallel
        # Stack all expert up projections
        expert_up_weights = torch.stack([self.expert_up_proj[i].weight for i in range(self.num_experts)], dim=0)
        expert_down_weights = torch.stack([self.expert_down_proj[i].weight for i in range(self.num_experts)], dim=0)
        
        # Compute expert outputs for all tokens and experts
        # [num_experts, n_embd, specialized_dim]
        x_expanded = x_flat.unsqueeze(0).expand(self.num_experts, -1, -1)  # [num_experts, batch_size * seq_len, n_embd]
        
        # Batch matrix multiply: [num_experts, batch_size * seq_len, specialized_dim]
        expert_hidden = torch.bmm(x_expanded, expert_up_weights.transpose(1, 2))
        expert_hidden = self.activation(expert_hidden)
        
        # Batch matrix multiply: [num_experts, batch_size * seq_len, n_embd]
        expert_outputs = torch.bmm(expert_hidden, expert_down_weights.transpose(1, 2))
        
        # Combine expert outputs according to routing weights
        # [batch_size * seq_len, num_experts, n_embd]
        expert_outputs = expert_outputs.permute(1, 0, 2)
        
        # Apply routing weights
        # expert_one_hot: [batch_size * seq_len, experts_per_token, num_experts]
        # expert_weights_flat: [batch_size * seq_len, experts_per_token]
        weighted_expert_mask = expert_one_hot * expert_weights_flat.unsqueeze(-1)  # [batch_size * seq_len, experts_per_token, num_experts]
        combined_weights = weighted_expert_mask.sum(dim=1)  # [batch_size * seq_len, num_experts]
        
        # Apply weights to expert outputs
        expert_output = torch.einsum('bn,bne->be', combined_weights, expert_outputs)  # [batch_size * seq_len, n_embd]
        
        # Reshape back to original shape
        expert_output = expert_output.view(batch_size, seq_len, n_embd)
        
        # Combine shared and expert-specific outputs
        return shared_output + expert_output
    
    def _init_weights(self):
        """Initialize weights with scaled initialization."""
        scale = 1.0 / math.sqrt(self.config.n_embd)
        
        # Shared weights
        nn.init.normal_(self.shared_up_proj.weight, mean=0.0, std=scale)
        nn.init.normal_(self.shared_down_proj.weight, mean=0.0, std=scale)
        
        # Expert weights (smaller initialization for stability)
        expert_scale = scale * 0.1
        for expert_id in range(self.num_experts):
            nn.init.normal_(self.expert_up_proj[expert_id].weight, mean=0.0, std=expert_scale)
            nn.init.normal_(self.expert_down_proj[expert_id].weight, mean=0.0, std=expert_scale)


class Router(nn.Module):
    """
    Router for selecting experts.
    """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        
        # Router network
        self.router = nn.Linear(config.n_embd, config.num_experts, bias=False)
        
        # Router temperature
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        # Loss coefficients
        self.router_z_loss_coef = 0.01
        self.load_balance_coef = 0.01
        
        # Initialize
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.
        
        Returns:
            expert_indices: Selected expert indices [batch_size, seq_len, experts_per_token]
            expert_weights: Normalized weights [batch_size, seq_len, experts_per_token]
            router_loss: Auxiliary loss for load balancing
        """
        # Compute router logits
        router_logits = self.router(x) / (self.temperature.abs() + 1e-6)
        
        # Get top-k experts
        routing_weights = F.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.topk(
            routing_weights, self.experts_per_token, dim=-1
        )
        
        # Normalize expert weights
        expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Compute auxiliary losses
        # Load balancing loss
        actual_load = routing_weights.mean(dim=[0, 1])
        ideal_load = torch.ones_like(actual_load) / self.num_experts
        load_balance_loss = F.kl_div(
            actual_load.log(),
            ideal_load,
            reduction='batchmean',
            log_target=False
        )
        
        # Router z-loss (encourages exploration)
        router_z_loss = torch.square(router_logits).mean()
        
        # Total router loss
        router_loss = (
            self.router_z_loss_coef * router_z_loss +
            self.load_balance_coef * load_balance_loss
        )
        
        return expert_indices, expert_weights, router_loss


class MOEMLABlock(nn.Module):
    """
    MOE-MLA block combining MLA attention and MoE feed-forward network.
    """
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        
        # Layer normalization
        if config.use_dyt:
            self.norm1 = DynamicTanh(config.n_embd, alpha_init=config.dyt_alpha_init)
            self.norm2 = DynamicTanh(config.n_embd, alpha_init=config.dyt_alpha_init)
        else:
            self.norm1 = RMSNorm(config.n_embd)
            self.norm2 = RMSNorm(config.n_embd)
        
        # Multi-head Latent Attention
        if config.use_fp8:
            self.attn = MLA_FP8(config)
        else:
            self.attn = MLA(config)
        
        # Router for MoE
        self.router = Router(config)
        
        # Shared expert MLP
        self.ffn = SharedExpertMLP(config, config.num_experts)
        
        # Gradient checkpointing
        self.use_checkpoint = config.use_gradient_checkpointing
    
    def _attn_block(self, x, start_pos=0, freqs_cis=None, mask=None):
        return self.attn(self.norm1(x), start_pos, freqs_cis, mask)
    
    def _ffn_block(self, x):
        normalized = self.norm2(x)
        
        # Route tokens to experts
        expert_indices, expert_weights, router_loss = self.router(normalized)
        
        # Process through MoE FFN
        output = self.ffn(normalized, expert_indices, expert_weights)
        
        # Return both output and router_loss
        return output, router_loss
    
    def forward(self, x, start_pos=0, freqs_cis=None, mask=None):
        # Apply attention
        if self.use_checkpoint and self.training:
            attn_output = checkpoint.checkpoint(
                self._attn_block, x, start_pos, freqs_cis, mask,
                use_reentrant=False
            )
        else:
            attn_output = self._attn_block(x, start_pos, freqs_cis, mask)
        
        # Handle FP8 conversion if needed
        if attn_output.dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and x.dtype != attn_output.dtype:
            attn_output = attn_output.to(x.dtype)
        
        x = x + attn_output
        
        # Apply feed-forward network
        if self.use_checkpoint and self.training:
            # For checkpointing, we need to handle the router loss separately
            ffn_output, router_loss = checkpoint.checkpoint(
                self._ffn_block, x,
                use_reentrant=False
            )
            # Store router loss outside of checkpoint
            self.last_router_loss = router_loss
        else:
            ffn_output, router_loss = self._ffn_block(x)
            self.last_router_loss = router_loss
        
        # Handle FP8 conversion if needed
        if ffn_output.dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and x.dtype != ffn_output.dtype:
            ffn_output = ffn_output.to(x.dtype)
        
        x = x + ffn_output
        
        return x


class MOEMLA(nn.Module):
    """
    MOE-MLA Model: Mixture of Experts with Multi-head Latent Attention.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([MOEMLABlock(config, i) for i in range(config.n_layer)]),
            ln_f = DynamicTanh(config.n_embd, alpha_init=config.dyt_alpha_init) if config.use_dyt else RMSNorm(config.n_embd)
        ))
        
        # LM head (weight tied with embeddings)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        
        # Router loss tracking
        self.last_router_losses = []
        
        # Rotary embeddings setup
        self._setup_rope()
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Parameter count
        self.param_count = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {self.param_count/1e6:.2f}M")
    
    def _setup_rope(self):
        """Setup rotary positional embeddings."""
        rope_theta = self.config.rope_theta
        max_seq_len = self.config.block_size
        head_dim = self.config.qk_rope_head_dim
        
        # Compute rotary frequencies
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Apply rope scaling if provided
        if self.config.rope_scaling is not None:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            
            if scaling_type == "linear":
                self.register_buffer("freqs_cis", precompute_freqs_cis_with_linear_scaling(
                    head_dim, max_seq_len, rope_theta, scaling_factor, 
                    self.config.original_max_seq_len
                ))
            else:
                raise ValueError(f"Unknown RoPE scaling type: {scaling_type}")
        else:
            self.register_buffer("freqs_cis", precompute_freqs_cis(
                head_dim, max_seq_len, rope_theta
            ))
    
    def _init_weights(self, module):
        """Initialize weights with scaled initialization."""
        if isinstance(module, nn.Linear):
            scale = 1.0 / math.sqrt(self.config.n_embd)
            nn.init.normal_(module.weight, mean=0.0, std=scale)
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_router_loss(self):
        """Get the accumulated router loss from all layers."""
        if not self.last_router_losses:
            return 0.0
        return sum(self.last_router_losses) / len(self.last_router_losses)
    
    def forward(self, idx, targets=None):
        # Ensure we're not caching during training
        if self.training:
            self._set_inference_mode(False)
        
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # Generate position indices
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # Forward the model
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        
        # Prepare attention mask
        mask = None
        if t > 1:
            mask = torch.full((t, t), float("-inf"), device=device, requires_grad=False).triu_(1)
        
        # Extract rotary embeddings
        freqs_cis = self.freqs_cis[:t].detach()
        
        # Process through layers and collect router losses
        self.last_router_losses = []
        for block in self.transformer.h:
            x = block(x, 0, freqs_cis, mask)
            if hasattr(block, 'last_router_loss'):
                self.last_router_losses.append(block.last_router_loss)
        
        # Final normalization
        x = self.transformer.ln_f(x)
        
        # Compute logits and loss
        if targets is not None:
            logits = self.lm_head(x)
            
            # Language modeling loss
            if self.config.label_smoothing > 0.0:
                # Apply label smoothing
                num_classes = logits.size(-1)
                smoothing = self.config.label_smoothing
                
                confidence = 1.0 - smoothing
                smoothing_value = smoothing / (num_classes - 1)
                
                with torch.no_grad():
                    true_dist = torch.zeros_like(logits)
                    true_dist.fill_(smoothing_value)
                    true_dist.scatter_(-1, targets.unsqueeze(-1), confidence)
                
                log_probs = F.log_softmax(logits.view(-1, logits.size(-1)), dim=-1)
                loss = -(true_dist.view(-1, true_dist.size(-1)) * log_probs).sum(-1)
                
                with torch.no_grad():
                    mask = (targets != -1).float()
                loss = (loss * mask.view(-1)).sum() / mask.sum()
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
            # Add router loss for load balancing
            router_loss = self.get_router_loss()
            # Always add router_loss to avoid data-dependent branching
            loss = loss + router_loss
            
        else:
            # Inference: only compute logits for last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss
    
    def _set_inference_mode(self, use_inference=True):
        """Set the MLA blocks to inference mode for KV caching."""
        try:
            if use_inference:
                self.eval()
            else:
                self.train()
            
            for name, module in self.named_modules():
                if hasattr(module, 'set_inference_mode') and module is not self:
                    try:
                        module.set_inference_mode(use_inference)
                    except Exception as e:
                        print(f"Error setting inference mode for module {name}: {e}")
        except Exception as e:
            print(f"Error setting inference mode: {e}")
    
    def clear_cache(self):
        """Clear all KV caches in MLA attention layers."""
        for name, module in self.named_modules():
            if isinstance(module, (MLA, MLA_FP8)):
                module.set_inference_mode(False)
                module.k_cache = None
                module.v_cache = None
                module.kv_cache = None
                module.pe_cache = None
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=None, temperature=1.0, top_k=None, prompt=None, gen_length=None):
        """Generate text autoregressively."""
        # Handle compatibility
        if prompt is not None:
            idx = prompt
        
        tokens_to_generate = max_new_tokens
        if tokens_to_generate is None and gen_length is not None:
            tokens_to_generate = gen_length
        if tokens_to_generate is None:
            tokens_to_generate = 20
        
        self.eval()
        
        for _ in range(tokens_to_generate):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        result = (idx, None)
        self._set_inference_mode(False)
        
        return result
    
    def estimate_mfu(self, batch_size, dt, seq_length=None):
        """Estimate model FLOPS utilization (MFU) in percentage."""
        if seq_length is None:
            seq_length = self.config.block_size
            
        return utils_estimate_mfu(
            model=self,
            batch_size=batch_size,
            seq_length=seq_length,
            dt=dt,
            dtype=torch.bfloat16 if not self.config.use_fp8 else torch.float16
        )
    
    def set_gradient_checkpointing(self, value: bool):
        """Set gradient checkpointing for all transformer blocks."""
        self.config.use_gradient_checkpointing = value
        for block in self.transformer.h:
            block.use_checkpoint = value
    
    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, optimizer_type=None, **kwargs):
        """Configure optimizer with GaLore2 support."""
        from models.optimizers import configure_optimizer_for_gpt
        
        # Default to AdamW if no optimizer type specified
        if optimizer_type is None:
            optimizer_type = "adamw"
            print("No optimizer type specified, defaulting to AdamW")
        
        # Extract GaLore configuration from kwargs if present
        galore_config = None
        galore_quantize_proj = None
        if optimizer_type in ["galore", "galore-8bit", "galore2"]:
            galore_config = {
                "rank": kwargs.get("galore_rank", 128),
                "update_proj_gap": kwargs.get("galore_update_proj_gap", 200),
                "scale": kwargs.get("galore_scale", 0.25),
                "proj_type": kwargs.get("galore_proj_type", "std")
            }
            if optimizer_type == "galore2":
                galore_quantize_proj = kwargs.get("galore_quantize_proj", None)
        
        # Use the GPT optimizer configuration
        optimizer = configure_optimizer_for_gpt(
            model=self,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
            device_type=device_type,
            optimizer_type=optimizer_type,
            galore_config=galore_config,
            galore_quantize_proj=galore_quantize_proj
        )
        
        print(f"Configured {optimizer_type} optimizer for MOE-MLA model")
        return optimizer


def precompute_freqs_cis(dim, max_seq_len, theta):
    """Precompute the frequency tensor for complex exponentials (RoPE)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    
    # Complex exponentials
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def precompute_freqs_cis_with_linear_scaling(dim, max_seq_len, theta, scaling_factor, original_max_seq_len):
    """Precompute rotary embeddings with linear scaling for extended context."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    
    # Scale frequencies for extended context
    if max_seq_len > original_max_seq_len:
        scale_factor = 1.0 / scaling_factor
        freqs = freqs * scale_factor
    
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    
    # Complex exponentials
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def create_moe_mla_model(
    size: str = 'small',
    n_layer: Optional[int] = None,
    n_embd: Optional[int] = None,
    n_head: Optional[int] = None,
    num_experts: Optional[int] = None,
    experts_per_token: Optional[int] = None,
    shared_weight_ratio: Optional[float] = None,
    vocab_size: int = 50304,
    block_size: int = 4096,
    dropout: float = 0.0,
    use_fp8: bool = False,
    use_dyt: bool = False,
    **kwargs
):
    """
    Create a MOE-MLA model with predefined sizes.
    
    Args:
        size: Model size ('small', 'medium', 'large', 'xl')
        n_layer: Override number of layers
        n_embd: Override embedding dimension
        n_head: Override number of heads
        num_experts: Number of experts (default: 8)
        experts_per_token: Experts selected per token (default: 2)
        shared_weight_ratio: Ratio of shared weights (default: 0.75)
        vocab_size: Vocabulary size
        block_size: Maximum sequence length
        dropout: Dropout probability
        use_fp8: Whether to use FP8 precision
        use_dyt: Whether to use Dynamic Tanh normalization
        **kwargs: Additional configuration arguments
    
    Returns:
        MOEMLA: Configured model
    """
    # Define model sizes
    sizes = {
        'small': {
            'n_layer': 12,
            'n_embd': 768,
            'n_head': 12,
        },
        'medium': {
            'n_layer': 24,
            'n_embd': 1024,
            'n_head': 16,
        },
        'large': {
            'n_layer': 32,
            'n_embd': 2048,
            'n_head': 16,
        },
        'xl': {
            'n_layer': 40,
            'n_embd': 2560,
            'n_head': 20,
        }
    }
    
    # Start with the selected size
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
    
    # Add MOE-specific parameters
    config_dict.update({
        'num_experts': num_experts or 8,
        'experts_per_token': experts_per_token or 2,
        'shared_weight_ratio': shared_weight_ratio or 0.75,
        'vocab_size': vocab_size,
        'block_size': block_size,
        'dropout': dropout,
        'use_fp8': use_fp8,
        'use_dyt': use_dyt,
    })
    
    # Add any additional parameters
    config_dict.update(kwargs)
    
    # Create config and model
    config = MOEMLAConfig(**config_dict)
    model = MOEMLA(config)
    
    return model