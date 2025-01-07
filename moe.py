import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import custom_fwd, custom_bwd
import torch.utils.checkpoint as checkpoint
from model import CausalSelfAttention
from typing import Optional
import inspect
import gc

class Router(nn.Module):
    """
    Router module that determines which expert should process each token.
    Uses top-k routing with load balancing.
    """
    def __init__(self, input_dim: int, num_experts: int, k: int = 2, capacity_factor: float = 1.25):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor
        
        # Router projection
        self.router = nn.Linear(input_dim, num_experts, bias=False)
        
        # Initialize with smaller weights for better stability
        nn.init.normal_(self.router.weight, mean=0.0, std=0.001)
        
        # Pre-register buffers with fixed size
        max_batch_size = 32  # Adjust this based on your needs
        max_seq_len = 256
        self.register_buffer('_indices_buffer', 
            torch.arange(max_batch_size * max_seq_len).unsqueeze(1).expand(-1, self.k),
            persistent=False)
        self.register_buffer('_zeros_buffer', 
            torch.zeros((self.num_experts,)),
            persistent=False)
        self.register_buffer('_dispatch_buffer', 
            torch.zeros((max_batch_size * max_seq_len, self.num_experts)),
            persistent=False)
        self.register_buffer('_ones_buffer',
            torch.ones((max_batch_size * max_seq_len,), dtype=torch.float32),
            persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        combined_batch_size = batch_size * seq_len
        device = x.device
        
        # Use slices of pre-allocated buffers
        indices = self._indices_buffer[:combined_batch_size]
        dispatch_mask = self._dispatch_buffer[:combined_batch_size]
        dispatch_mask.zero_()
        expert_counts = self._zeros_buffer
        expert_counts.zero_()
        ones = self._ones_buffer[:combined_batch_size]
        
        # Calculate expert capacity
        capacity = int(self.capacity_factor * combined_batch_size * self.k / self.num_experts)
        
        # Compute router logits and probabilities in one go
        router_logits = self.router(x.view(combined_batch_size, -1))
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Get and normalize top-k in one operation
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Fused routing operation
        for k_idx in range(self.k):
            expert_idx = top_k_indices[:, k_idx]
            mask = expert_counts[expert_idx] < capacity
            expert_counts.scatter_add_(0, expert_idx[mask], ones[mask])
            dispatch_mask[indices[mask, k_idx], expert_idx[mask]] = top_k_weights[mask, k_idx]
        
        # Handle unrouted tokens efficiently
        unrouted = dispatch_mask.sum(dim=-1) == 0
        if torch.any(unrouted):
            least_loaded = (capacity - expert_counts).argmax()
            dispatch_mask[unrouted, least_loaded] = 1.0
        
        # Renormalize dispatch mask (in-place)
        dispatch_mask = dispatch_mask / (dispatch_mask.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Store routing weights for loss computation
        routing_weights_detached = routing_weights.detach()
        
        # Compute losses efficiently using detached tensors to avoid double backward
        router_z_loss = torch.mean(torch.square(router_logits))
        expert_counts = dispatch_mask.sum(0)
        target_count = combined_batch_size * self.k / self.num_experts
        load_balance_loss = torch.mean(torch.square(expert_counts / combined_batch_size - target_count / combined_batch_size))
        
        router_loss = 0.001 * router_z_loss + 0.001 * load_balance_loss
        
        return routing_weights_detached, dispatch_mask, router_loss

class ExpertMLP(nn.Module):
    """
    Expert MLP module with SwiGLU activation.
    Similar to the base MLP but with potential for specialization.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Use 4 * n_embd as hidden dimension by default
        hidden_dim = 4 * config.n_embd
        
        # Combined projections for efficiency
        self.gate_up_proj = nn.Linear(config.n_embd, 2 * hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights with special scale
        self._init_weights()

    def _init_weights(self):
        scale = 2 / (self.config.n_embd ** 0.5)
        nn.init.normal_(self.gate_up_proj.weight, mean=0.0, std=scale)
        if self.gate_up_proj.bias is not None:
            nn.init.zeros_(self.gate_up_proj.bias)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=scale)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Combined projection
        combined = self.gate_up_proj(x)
        
        # SwiGLU activation
        x1, x2 = combined.chunk(2, dim=-1)
        hidden = F.silu(x2) * x1
        
        # Final projection with dropout
        return self.dropout(self.down_proj(hidden))

class MoELayer(nn.Module):
    """
    Mixture of Experts layer that combines multiple expert MLPs with a router.
    """
    def __init__(self, config, num_experts: int = 8, k: int = 2, capacity_factor: float = 1.25):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.k = k
        
        # Create router
        self.router = Router(config.n_embd, num_experts, k, capacity_factor)
        
        # Create experts as a single large model for parallel computation
        self.experts = nn.ModuleList([ExpertMLP(config) for _ in range(num_experts)])
        
        # Layer norm before routing
        self.norm = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        batch_size, seq_len, hidden_dim = x.shape
        combined_batch_size = batch_size * seq_len
        
        # Normalize input
        normalized = self.norm(x)
        
        # Get routing weights and dispatch mask
        combine_weights, dispatch_mask, router_loss = self.router(normalized)
        
        # Reshape input for expert processing
        reshaped_input = normalized.view(-1, hidden_dim)
        
        # Process all experts in parallel
        # [num_experts, combined_batch_size, hidden_dim]
        expert_outputs = torch.stack([
            expert(reshaped_input) * dispatch_mask[:, i, None]
            for i, expert in enumerate(self.experts)
        ])
        
        # Sum expert outputs
        # [combined_batch_size, hidden_dim]
        combined_output = expert_outputs.sum(dim=0)
        
        # Reshape back to original dimensions
        output = combined_output.view(batch_size, seq_len, hidden_dim)
        
        return output, router_loss

class MoEBlock(nn.Module):
    """
    Transformer block that uses MoE instead of standard MLP.
    """
    def __init__(self, config, num_experts: int = 8, k: int = 2):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.moe = MoELayer(config, num_experts=num_experts, k=k)
        self.use_checkpoint = True

    def forward(self, x: torch.Tensor, key_value: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        # Attention
        attn_output = self.attn(self.ln_1(x), key_value=key_value)
        x = x + attn_output
        
        # MoE
        if self.use_checkpoint and self.training:
            moe_output, router_loss = checkpoint.checkpoint(
                self.moe,
                self.ln_2(x),
                use_reentrant=False
            )
        else:
            moe_output, router_loss = self.moe(self.ln_2(x))
        
        x = x + moe_output
        
        return x, router_loss

class MoEEncoderDecoderGPT(nn.Module):
    """
    Encoder-Decoder model that uses MoE blocks instead of standard transformer blocks.
    """
    def __init__(self, encoder_config, decoder_config, num_experts: int = 8, k: int = 2):
        super().__init__()
        
        # Ensure compatible dimensions
        assert encoder_config.n_embd == decoder_config.n_embd
        assert encoder_config.vocab_size == decoder_config.vocab_size
        assert encoder_config.block_size == decoder_config.block_size
        
        # Shared embeddings
        self.shared_embedding = nn.Embedding(encoder_config.vocab_size, encoder_config.n_embd)
        self.shared_pos_embedding = nn.Embedding(encoder_config.block_size, encoder_config.n_embd)
        
        # Create encoder and decoder with MoE blocks
        self.encoder = nn.ModuleDict(dict(
            drop = nn.Dropout(encoder_config.dropout),
            h = nn.ModuleList([MoEBlock(encoder_config, num_experts, k) for _ in range(encoder_config.n_layer)]),
            ln_f = nn.LayerNorm(encoder_config.n_embd)
        ))
        
        self.decoder = nn.ModuleDict(dict(
            drop = nn.Dropout(decoder_config.dropout),
            h = nn.ModuleList([MoEBlock(decoder_config, num_experts, k) for _ in range(decoder_config.n_layer)]),
            ln_f = nn.LayerNorm(decoder_config.n_embd)
        ))
        
        # Output head (shared with input embeddings)
        self.lm_head = nn.Linear(decoder_config.n_embd, decoder_config.vocab_size, bias=False)
        self.lm_head.weight = self.shared_embedding.weight
        
        # Cross-attention
        self.cross_attention = nn.ModuleList([
            CausalSelfAttention(decoder_config) 
            for _ in range(decoder_config.n_layer)
        ])
        
        self.cross_ln = nn.ModuleList([
            nn.LayerNorm(decoder_config.n_embd)
            for _ in range(decoder_config.n_layer)
        ])
        
        # Save configs
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, encoder_idx: torch.Tensor, decoder_idx: torch.Tensor, 
                targets: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass of the MoE encoder-decoder model.
        
        Args:
            encoder_idx: Input tokens for encoder [batch_size, encoder_seq_len]
            decoder_idx: Input tokens for decoder [batch_size, decoder_seq_len]
            targets: Optional target tokens [batch_size, decoder_seq_len]
            
        Returns:
            logits: Output logits
            loss: Optional loss value if targets provided
            router_loss: Combined routing loss from all MoE layers
        """
        device = encoder_idx.device
        
        # Ensure input has correct shape
        if encoder_idx.dim() == 1:
            encoder_idx = encoder_idx.unsqueeze(0)
        if decoder_idx.dim() == 1:
            decoder_idx = decoder_idx.unsqueeze(0)
        
        # Process encoder input
        encoder_pos = torch.arange(0, encoder_idx.size(1), device=device)
        encoder_emb = self.shared_embedding(encoder_idx)
        encoder_pos_emb = self.shared_pos_embedding(encoder_pos)
        
        # Adjust dimensions for broadcasting
        encoder_pos_emb = encoder_pos_emb.unsqueeze(0).expand(encoder_idx.size(0), -1, -1)
        
        x = self.encoder.drop(encoder_emb + encoder_pos_emb)
        
        total_router_loss = torch.tensor(0.0, device=device)
        
        # Encoder forward pass
        for block in self.encoder.h:
            x, router_loss = block(x)
            total_router_loss = total_router_loss + router_loss
            
        encoder_output = self.encoder.ln_f(x)
        
        # Process decoder input
        decoder_pos = torch.arange(0, decoder_idx.size(1), device=device)
        decoder_emb = self.shared_embedding(decoder_idx)
        decoder_pos_emb = self.shared_pos_embedding(decoder_pos)
        
        # Adjust dimensions for broadcasting
        decoder_pos_emb = decoder_pos_emb.unsqueeze(0).expand(decoder_idx.size(0), -1, -1)
        
        x = self.decoder.drop(decoder_emb + decoder_pos_emb)
        
        # Decoder forward pass with cross-attention
        for i, (block, cross_attn) in enumerate(zip(self.decoder.h, self.cross_attention)):
            # Self-attention
            attn_output = block.attn(block.ln_1(x))
            x = x + attn_output
            
            # Cross-attention
            cross_x = self.cross_ln[i](x)
            cross_output = cross_attn(cross_x, key_value=encoder_output)
            x = x + cross_output
            
            # MoE layer
            moe_input = block.ln_2(x)
            moe_out, router_loss = block.moe(moe_input)
            x = x + moe_out
            total_router_loss = total_router_loss + router_loss
            
        x = self.decoder.ln_f(x)
        
        # Calculate logits and loss
        if targets is not None:
            logits = self.lm_head(x)
            
            # Scale down the router loss by the number of MoE layers
            num_moe_layers = len(self.encoder.h) + len(self.decoder.h)
            avg_router_loss = total_router_loss / num_moe_layers
            
            # Calculate cross entropy loss using reshape instead of view
            logits_2d = logits.reshape(-1, logits.size(-1))
            targets_1d = targets.reshape(-1)
            
            ce_loss = F.cross_entropy(
                logits_2d,
                targets_1d,
                ignore_index=-1
            )
            
            # Combine losses with a very small coefficient for router loss
            loss = ce_loss + 0.0001 * avg_router_loss
        else:
            # For generation, only compute logits for the last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss, total_router_loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, 
                temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """
        Generate text using the MoE model.
        """
        # Ensure input has correct shape
        
        print(idx)
        
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        
        # Encode input sequence once
        encoder_pos = torch.arange(0, idx.size(1), device=idx.device)
        encoder_emb = self.shared_embedding(idx)
        encoder_pos_emb = self.shared_pos_embedding(encoder_pos)
        
        # Adjust dimensions for broadcasting
        encoder_pos_emb = encoder_pos_emb.unsqueeze(0).expand(idx.size(0), -1, -1)
        
        x = self.encoder.drop(encoder_emb + encoder_pos_emb)
        
        # Encoder forward pass
        for block in self.encoder.h:
            x, _ = block(x)
            
        encoder_output = self.encoder.ln_f(x)
        
        # Create prompt-aware initial state for decoder
        prompt_summary = torch.mean(encoder_output, dim=1, keepdim=True)  # [B, 1, C]
        
        # Initialize decoder input with BOS token + prompt
        bos_token = torch.full(
            (idx.size(0), 1),
            self.shared_embedding.weight.size(0)-2,  # BOS token
            dtype=torch.long,
            device=idx.device
        )
        # Feed the entire prompt to decoder 
        decoder_idx = torch.cat([bos_token, idx], dim=1)
        
        # Track generated tokens for repetition detection
        last_tokens = []
        repetition_penalty = 1.5  # Increased from 1.2
        min_tokens_before_repetition = 5  # Increased from 3
        
        # Generate tokens autoregressively
        for _ in range(max_new_tokens):
            # Ensure we don't exceed maximum sequence length
            if decoder_idx.size(1) > self.decoder_config.block_size:
                decoder_idx = decoder_idx[:, -self.decoder_config.block_size:]
            
            # Decoder forward pass
            decoder_pos = torch.arange(0, decoder_idx.size(1), device=decoder_idx.device)
            decoder_emb = self.shared_embedding(decoder_idx)
            decoder_pos_emb = self.shared_pos_embedding(decoder_pos)
            
            # Add prompt context to decoder embeddings
            decoder_emb = decoder_emb + 0.3 * prompt_summary.expand(-1, decoder_emb.size(1), -1)
            
            # Adjust dimensions for broadcasting
            decoder_pos_emb = decoder_pos_emb.unsqueeze(0).expand(decoder_idx.size(0), -1, -1)
            
            # Process decoder input
            decoder_x = self.decoder.drop(decoder_emb + decoder_pos_emb)
            
            # Decoder layers with cross-attention
            for i, (block, cross_attn) in enumerate(zip(self.decoder.h, self.cross_attention)):
                # Self-attention
                attn_output = block.attn(block.ln_1(decoder_x))
                decoder_x = decoder_x + attn_output
                
                # Cross-attention with encoder output
                cross_x = self.cross_ln[i](decoder_x)
                # Explicitly use encoder output in cross-attention
                cross_output = cross_attn(
                    cross_x, 
                    key_value=encoder_output,
                    is_generation=True
                )
                decoder_x = decoder_x + cross_output
                
                # MoE layer
                moe_input = block.ln_2(decoder_x)
                moe_out, _ = block.moe(moe_input)
                decoder_x = decoder_x + moe_out
            
            decoder_x = self.decoder.ln_f(decoder_x)
            
            # Get logits for next token
            logits = self.lm_head(decoder_x[:, -1:])
            logits = logits.reshape(decoder_x.size(0), -1)  # Use reshape instead of squeeze
            
            # Apply repetition penalty
            if len(last_tokens) >= min_tokens_before_repetition:
                for token in set(last_tokens[-min_tokens_before_repetition:]):
                    logits[:, token] = logits[:, token] / repetition_penalty
            
            # Apply temperature and top-k sampling
            logits = logits / (temperature + 1e-5)  # Add small epsilon for stability
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Add length penalty to discourage very short sequences
            if decoder_idx.size(1) < 10:  # Minimum desired length
                length_penalty = torch.ones_like(logits) * -0.5
                logits = logits + length_penalty
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Update last tokens
            last_tokens.append(next_token.item())
            if len(last_tokens) > min_tokens_before_repetition * 2:
                last_tokens.pop(0)
            
            # Stop if we generate an EOS token
            if (next_token == self.shared_embedding.weight.size(0)-1).any():
                break
            
            # Append to sequence
            decoder_idx = torch.cat((decoder_idx, next_token), dim=1)
        
        return decoder_idx

    def set_gradient_checkpointing(self, value: bool):
        """Set gradient checkpointing for all transformer blocks."""
        # Set checkpointing for encoder blocks
        for block in self.encoder.h:
            block.use_checkpoint = value
            
        # Set checkpointing for decoder blocks
        for block in self.decoder.h:
            block.use_checkpoint = value

    def estimate_mfu(self, batch_size: int, dt: float) -> float:
        """Estimate model flops utilization (MFU) in percentage."""
        # Encoder flops
        encoder_flops = batch_size * self.encoder_config.block_size * self.encoder_config.n_layer * (
            # Attention
            8 * self.encoder_config.n_embd * self.encoder_config.block_size +
            # MLP (assuming average expert utilization of k/num_experts)
            2 * 4 * self.encoder_config.n_embd * self.encoder_config.n_embd * (2/8)
        )
        
        # Decoder flops
        decoder_flops = batch_size * self.decoder_config.block_size * self.decoder_config.n_layer * (
            # Self-attention
            8 * self.decoder_config.n_embd * self.decoder_config.block_size +
            # Cross-attention
            8 * self.decoder_config.n_embd * self.encoder_config.block_size +
            # MLP (assuming average expert utilization of k/num_experts)
            2 * 4 * self.decoder_config.n_embd * self.decoder_config.n_embd * (2/8)
        )
        
        total_flops = encoder_flops + decoder_flops
        
        # Get device compute capability
        if hasattr(torch.cuda, 'get_device_capability'):
            device = next(self.parameters()).device
            if device.type == 'cuda':
                gpu_flops = {
                    (7, 0): 14e12,    # A100
                    (8, 0): 19e12,    # A100
                    (8, 6): 36e12,    # H100
                    (9, 0): 39e12     # H200
                }.get(torch.cuda.get_device_capability(device), 14e12)  # Default to A100
            else:
                gpu_flops = 14e12  # Default to A100 if not on GPU
        else:
            gpu_flops = 14e12  # Default if can't detect
        
        mfu = total_flops / (dt * gpu_flops)
        return mfu 

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure the AdamW optimizer with weight decay.
        Handles parameter groups for experts and non-expert parameters separately.
        """
        # Collect all parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Create parameter groups
        # 1. Expert parameters (MLP weights in experts)
        # 2. Router parameters (need lower learning rate)
        # 3. Other parameters (attention, embeddings, etc.)
        expert_params = []
        router_params = []
        other_params = []
        
        for name, param in param_dict.items():
            if 'experts' in name:
                expert_params.append(param)
            elif 'router' in name:
                router_params.append(param)
            else:
                other_params.append(param)
        
        # Create optimizer groups with different hyperparameters
        optim_groups = [
            # Expert parameters
            {
                'params': expert_params,
                'weight_decay': weight_decay,
                'lr': learning_rate
            },
            # Router parameters (lower learning rate)
            {
                'params': router_params,
                'weight_decay': 0.0,  # No weight decay for router
                'lr': learning_rate * 0.1  # Lower learning rate for stability
            },
            # Other parameters
            {
                'params': other_params,
                'weight_decay': weight_decay,
                'lr': learning_rate
            }
        ]
        
        # Print parameter counts
        num_expert_params = sum(p.numel() for p in expert_params)
        num_router_params = sum(p.numel() for p in router_params)
        num_other_params = sum(p.numel() for p in other_params)
        
        print(f"Expert parameters: {num_expert_params:,}")
        print(f"Router parameters: {num_router_params:,}")
        print(f"Other parameters: {num_other_params:,}")
        
        # Create AdamW optimizer with fused implementation if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra_args
        )
        
        print(f"Using fused AdamW: {use_fused}")
        
        return optimizer 