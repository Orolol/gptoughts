import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import custom_fwd, custom_bwd
import torch.utils.checkpoint as checkpoint
from typing import Optional, Tuple, List, Dict, Any
import inspect
import gc
import time
from contextlib import nullcontext
from dataclasses import dataclass

@dataclass
class LLaDAConfig:
    """Configuration class for LLaDA model hyperparameters."""
    
    block_size: int = 1024  # Maximum sequence length
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64
    n_layer: int = 12       # Number of transformer layers
    n_head: int = 12        # Number of attention heads
    n_embd: int = 768      # Embedding dimension
    dropout: float = 0.0    # Dropout probability
    bias: bool = True       # Whether to use bias in Linear and LayerNorm layers
    ratio_kv: int = 4      # Ratio of key/value heads
    mask_token_id: int = 126336  # Special token ID for [MASK]
    num_experts: int = 8    # Number of experts in the MoE layer
    k: int = 2             # Top-k experts to route tokens to
    temperature: float = 0.0  # Temperature for sampling during generation
    remasking: str = 'low_confidence'  # Remasking strategy: 'low_confidence' or 'random'
    use_checkpoint: bool = False  # Whether to use gradient checkpointing
    
class LLaDARouter(nn.Module):
    """
    Router module that determines which expert should process each token.
    Modified for LLaDA with improved stability.
    """
    def __init__(self, input_dim: int, num_experts: int, k: int = 2, capacity_factor: float = 1.5):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        # Ensure k is not larger than num_experts
        self.k = min(k, num_experts)
        self.capacity_factor = capacity_factor
        
        # Router network with normalization
        self.router = nn.Sequential(
            nn.Linear(input_dim, num_experts, bias=False),
            nn.LayerNorm(num_experts)
        )
        
        # Temperature parameter for routing
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        # Loss coefficients
        self.router_z_loss_coef = 0.01
        self.load_balance_coef = 0.01

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        # Compute router logits with memory optimization
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            try:
                # Compute router logits
                router_logits = self.router(x.reshape(-1, self.input_dim))
                
                # Scale logits by temperature
                router_logits = router_logits / (self.temperature.abs() + 1e-6)
                
                # Compute routing probabilities
                routing_weights = F.softmax(router_logits, dim=-1)
                
                # Get top-k routing weights and indices
                # Clamp k to ensure it's not larger than num_experts
                effective_k = min(self.k, self.num_experts)
                
                # Safety check to avoid empty or broken tensors
                if effective_k <= 0 or routing_weights.shape[-1] <= 0:
                    print(f"Warning: Invalid dimensions for topk: effective_k={effective_k}, dim size={routing_weights.shape[-1]}")
                    # Create fallback uniform weights
                    routing_weights_fallback = torch.ones((batch_size * seq_len, self.num_experts), 
                                                         device=x.device) / self.num_experts
                    dispatch_mask = routing_weights_fallback.view(batch_size, seq_len, -1)
                    router_loss = torch.tensor(0.0, device=x.device)
                    return routing_weights_fallback, dispatch_mask, router_loss
                
                # Ensure we don't request more elements than exist
                effective_k = min(effective_k, routing_weights.shape[-1])
                top_k_weights, top_k_indices = torch.topk(routing_weights, effective_k, dim=-1)
                
                # Normalize top-k weights
                top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)
                
                # Create dispatch mask with proper error handling
                dispatch_mask = torch.zeros_like(routing_weights)
                # Safely limit indices to avoid out-of-bounds errors
                safe_indices = torch.clamp(top_k_indices, 0, self.num_experts - 1)
                
                try:
                    # Use batched scatter for better performance
                    batch_indices = safe_indices
                    batch_src = top_k_weights
                    
                    # Ensure indices are within bounds one more time
                    if batch_indices.max() < routing_weights.shape[-1]:
                        dispatch_mask.scatter_(
                            dim=-1,
                            index=batch_indices,
                            src=batch_src
                        )
                    else:
                        # Fallback to uniform distribution if indices are out of bounds
                        print(f"Warning: Index {batch_indices.max().item()} out of bounds for dimension of size {routing_weights.shape[-1]}")
                        dispatch_mask = torch.ones_like(routing_weights) / self.num_experts
                except Exception as e:
                    # Fallback in case of errors - distribute evenly
                    dispatch_mask = torch.ones_like(routing_weights) / self.num_experts
                    print(f"Warning: Router scatter failed, using uniform distribution: {e}")
                
                # Compute load balancing loss
                ideal_load = torch.ones_like(routing_weights.mean(0)) / self.num_experts
                actual_load = routing_weights.mean(0)
                load_balance_loss = F.kl_div(
                    actual_load.log(),
                    ideal_load,
                    reduction='batchmean',
                    log_target=False
                )
                
                # Router z-loss to encourage exploration
                router_z_loss = torch.square(router_logits).mean()
                
                # Combine losses
                router_loss = (
                    self.router_z_loss_coef * router_z_loss +
                    self.load_balance_coef * load_balance_loss
                )
                
                # Reshape dispatch mask for batch processing
                dispatch_mask = dispatch_mask.view(batch_size, seq_len, -1)
                
                # Free memory for intermediate tensors
                del router_logits, top_k_weights, top_k_indices, safe_indices
                
                return routing_weights.detach(), dispatch_mask, router_loss
                
            except Exception as e:
                # Complete fallback in case of unexpected errors
                print(f"Router failed with error, using uniform routing: {e}")
                fallback_weights = torch.ones((batch_size * seq_len, self.num_experts), 
                                            device=x.device) / self.num_experts
                fallback_mask = fallback_weights.view(batch_size, seq_len, -1)
                fallback_loss = torch.tensor(0.0, device=x.device)
                return fallback_weights, fallback_mask, fallback_loss

class LLaDAExpert(nn.Module):
    """
    Expert module for LLaDA with shared parameters and adaptation layers.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Dimensions
        self.hidden_dim = 2 * config.n_embd
        
        # Shared parameters
        self.shared_up = nn.Linear(config.n_embd, self.hidden_dim, bias=config.bias)
        self.shared_gate = nn.Linear(config.n_embd, self.hidden_dim, bias=config.bias)
        self.shared_down = nn.Linear(self.hidden_dim, config.n_embd, bias=config.bias)
        
        # Adaptation dimensions
        self.adapt_dim = self.hidden_dim // 16
        self.pre_adapt = nn.Linear(config.n_embd, self.adapt_dim, bias=config.bias)
        self.post_adapt = nn.Linear(self.hidden_dim, self.adapt_dim, bias=config.bias)
        
        # Normalization and projections
        self.adapt_norm = nn.LayerNorm(self.adapt_dim)
        self.adapt_proj = nn.Linear(self.adapt_dim, self.hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        
        # Main computation with shared weights
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            up = self.shared_up(x)
            gate = self.shared_gate(x)
            hidden = F.silu(gate) * up
            
            # Free memory
            del gate, up
            
            # Adaptation pathway
            adapt_in = self.adapt_norm(self.pre_adapt(x))
            adapt_out = self.adapt_norm(self.post_adapt(hidden))
            
            # Compute adaptation weights
            adapt_weights = torch.bmm(adapt_in, adapt_out.transpose(1, 2))
            adapt_weights = torch.clamp(adapt_weights, -5.0, 5.0)
            adapt_weights = F.silu(adapt_weights)
            
            # Free memory
            del adapt_out
            
            # Apply adaptation with memory optimization
            adapt = torch.bmm(adapt_weights, adapt_in)
            
            # Free memory
            del adapt_weights, adapt_in
            
            adapt = self.adapt_proj(adapt)
            
            # Scale adaptation contribution
            adapt = 0.1 * adapt
            
            # Combine paths
            hidden = hidden + adapt
            
            # Free memory
            del adapt
            
            out = self.dropout(self.shared_down(hidden))
            
            # Free memory
            del hidden
            
            return out

    def _init_weights(self):
        scale = 1 / (self.config.n_embd ** 0.5)
        for layer in [self.shared_up, self.shared_gate, self.shared_down]:
            nn.init.normal_(layer.weight, mean=0.0, std=scale)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        
        adapt_scale = scale * 0.01
        for layer in [self.pre_adapt, self.post_adapt, self.adapt_proj]:
            nn.init.normal_(layer.weight, mean=0.0, std=adapt_scale)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.zeros_(layer.bias)

class LLaDAExpertGroup(nn.Module):
    """
    A group of experts for LLaDA with shared parameters.
    """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.config = config
        
        # Define dimensions
        self.hidden_dim = 2 * config.n_embd
        self.adapt_dim = self.hidden_dim // 16
        
        # Shared MLP backbone
        self.shared_mlp = LLaDAExpert(config)
        
        # Expert-specific adaptation
        self.expert_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.adapt_dim, self.adapt_dim, bias=False),
                nn.LayerNorm(self.adapt_dim)
            ) for _ in range(config.num_experts)
        ])
        
        # Projections
        self.expert_proj = nn.Linear(self.adapt_dim, self.hidden_dim, bias=False)
        self.output_proj = nn.Linear(self.hidden_dim, config.n_embd, bias=False)
        
        # Enable parallel computation
        self.parallel_adapters = True
        
        # Initialize with small values
        adapt_scale = 0.01 / (self.adapt_dim ** 0.5)
        for adapter in self.expert_adapters:
            nn.init.normal_(adapter[0].weight, mean=0.0, std=adapt_scale)
        nn.init.normal_(self.expert_proj.weight, mean=0.0, std=adapt_scale)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=adapt_scale)

    def forward(self, x: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        
        # Get shared representations
        shared_output = self.shared_mlp(x)
        
        if self.parallel_adapters:
            # Process all experts in parallel with better memory management
            expert_outputs = []
            masks = []
            
            # Ensure expert_weights has correct shape
            if expert_weights.dim() == 2:
                expert_weights = expert_weights.view(batch_size, seq_len, -1)
            
            # Safety check for expert_weights dimensions
            if expert_weights.shape[2] < self.num_experts:
                print(f"Warning: expert_weights has fewer experts ({expert_weights.shape[2]}) than expected ({self.num_experts})")
                # Pad with zeros to match expected number of experts
                padding = torch.zeros((batch_size, seq_len, self.num_experts - expert_weights.shape[2]), 
                                     device=expert_weights.device)
                expert_weights = torch.cat([expert_weights, padding], dim=2)
            
            # Create a combined output tensor to avoid memory fragmentation
            combined_output = torch.zeros_like(shared_output)
            
            # Process each expert with explicit memory management
            for i in range(self.num_experts):
                try:
                    # Safety check for out-of-bounds indexing
                    if i >= expert_weights.shape[2]:
                        continue
                        
                    mask = expert_weights[:, :, i] > 0
                    if mask.any():
                        # Process only the necessary tokens for this expert
                        expert_x = x[mask]
                        
                        if expert_x.numel() == 0:  # Skip if no tokens were selected
                            continue
                            
                        # Process through adapter with memory optimization
                        with torch.amp.autocast(enabled=True, device_type='cuda'):
                            # Process through adapter and projections in smaller chunks if needed
                            expert_hidden = self.shared_mlp.pre_adapt(expert_x)
                            expert_hidden = self.expert_adapters[i](expert_hidden)
                            expert_hidden = self.expert_proj(expert_hidden)
                            expert_output = self.output_proj(expert_hidden)
                            
                            # Free memory
                            del expert_hidden
                            
                            # Apply directly to the output tensor
                            combined_output[mask] = expert_output
                            
                            # Free memory for this iteration
                            del expert_x, expert_output
                            torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Warning: Error processing expert {i}: {e}")
                    # Continue with next expert on error
                    continue
            
            # Scale the expert contribution and add to shared output
            return shared_output + 0.1 * combined_output
        
        return shared_output

class LLaDAAttention(nn.Module):
    """
    Self-attention module for LLaDA with bidirectional attention (no causal masking).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Ensure embedding dimension is divisible by number of heads
        assert config.n_embd % config.n_head == 0
        
        # Key, query, value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Attention params
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Head size
        self.head_size = config.n_embd // config.n_head

    def forward(self, x):
        batch_size, seq_len, n_embd = x.shape
        
        # Calculate query, key, values
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape to multiple heads
        k = k.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2)
        
        # Use PyTorch's scaled dot-product attention with memory optimization
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            # Use SDPA for efficient attention computation
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,  # No causal mask here (bidirectional attention)
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )
            
            # Free memory
            del q, k, v
        
        # Reshape back to original
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y

class LLaDABlock(nn.Module):
    """
    Transformer block for LLaDA combining attention and MoE layers.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = LLaDAAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        
        # MoE components
        self.router = LLaDARouter(config.n_embd, config.num_experts, config.k)
        self.expert_group = LLaDAExpertGroup(config)
        
        # Router loss coefficient
        self.router_z_loss = 0.001
        
        # Gradient checkpointing flag
        self.use_checkpoint = False

    def forward(self, x):
        # Process in smaller chunks if needed
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            # Self-attention (bidirectional, not causal)
            residual = x
            attn_out = self.attn(self.ln_1(x))
            x = residual + attn_out
            
            # Free memory
            del residual, attn_out
            
            # MoE layer
            residual = x
            moe_input = self.ln_2(x)
            routing_weights, dispatch_mask, router_loss = self.router(moe_input)
            
            # Ensure dispatch_mask has correct shape
            if dispatch_mask.dim() == 2:
                batch_size, seq_len = x.shape[:2]
                dispatch_mask = dispatch_mask.view(batch_size, seq_len, -1)
                
            expert_output = self.expert_group(moe_input, dispatch_mask)
            
            # Free memory for MoE intermediates
            del moe_input, routing_weights, dispatch_mask
            
            x = residual + expert_output
            
            # Free memory
            del residual, expert_output
            
            return x, router_loss

class LLaDAModel(nn.Module):
    """
    LLaDA model combining MoE architecture with diffusion-based language modeling.
    """
    # Class-level warning counter
    _pos_warning_counter = 0
    _pos_warning_max = 5
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([LLaDABlock(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Output head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie token embedding weights with lm_head 
        self.lm_head.weight = self.tok_emb.weight
        
        # Dropout
        self.drop = nn.Dropout(config.dropout)
        
        # Special token ID for masking
        self.mask_token_id = config.mask_token_id
        # Check if mask_token_id is outside vocab range and warn
        if self.mask_token_id >= config.vocab_size:
            print(f"Warning: mask_token_id ({self.mask_token_id}) is outside the vocabulary range (0-{config.vocab_size-1}).")
            print(f"Will use vocab_size-1 ({config.vocab_size-1}) as a fallback during masking.")
        
        # Gradient checkpointing flag - enable by default for memory efficiency
        self.use_checkpoint = getattr(config, 'use_checkpoint', True)
        
        # Apply weight initialization
        self.apply(self._init_weights)
        
        # Timing stats for profiling
        self.timing_stats = None

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

    def set_timing_stats(self, timing_stats):
        """Set the timing stats object for profiling"""
        self.timing_stats = timing_stats

    def forward_process(self, input_ids, eps=1e-3):
        """Apply random masking with varying ratios for diffusion process"""
        batch_size, seq_len = input_ids.shape
        
        # Ensure mask_token_id is within vocabulary range
        safe_mask_token_id = min(self.mask_token_id, self.config.vocab_size - 1)
        
        # Sample random masking ratios between eps and 1-eps
        t = torch.rand(batch_size, device=input_ids.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, seq_len)
        
        # Apply masking randomly according to p_mask
        masked_indices = torch.rand((batch_size, seq_len), device=input_ids.device) < p_mask
        noisy_batch = torch.where(masked_indices, safe_mask_token_id, input_ids)
        
        return noisy_batch, masked_indices, p_mask

    def forward(self, input_ids, targets=None, apply_masking=True, eps=1e-3):
        """
        Forward pass through the model
        
        Args:
            input_ids: Tensor of token indices [batch_size, seq_len]
            targets: Optional tensor of target indices [batch_size, seq_len]
            apply_masking: Whether to apply mask to token embeddings
            eps: Small epsilon value for numerical stability
            
        Returns:
            logits: Tensor of logits [batch_size, seq_len, vocab_size]
            loss: Optional cross-entropy loss
            router_loss: Optional aux loss from router
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # Safety check: ensure all input IDs are within vocabulary range (strictly less than vocab_size)
        if torch.any(input_ids >= self.config.vocab_size):
            max_idx = torch.max(input_ids).item()
            print(f"Warning: Input contains token IDs outside vocabulary range. Max ID: {max_idx}, Vocab size: {self.config.vocab_size}")
            input_ids = torch.clamp(input_ids, 0, self.config.vocab_size - 1)
        
        # Also clamp targets if provided to avoid CUDA errors
        if targets is not None:
            if torch.any(targets >= self.config.vocab_size):
                max_target = torch.max(targets).item()
                print(f"Warning: Targets contain token IDs outside vocabulary range. Max ID: {max_target}, Vocab size: {self.config.vocab_size}")
                targets = torch.clamp(targets, 0, self.config.vocab_size - 1)
        
        # Apply masking if requested (for training)
        if apply_masking:
            noisy_batch, masked_indices, p_mask = self.forward_process(input_ids, eps)
        else:
            noisy_batch = input_ids
            masked_indices = None
            p_mask = None
        
        # Position embeddings - ensure position indices don't exceed embedding size
        max_pos = min(seq_len, self.config.block_size)
        pos = torch.arange(0, max_pos, device=device).unsqueeze(0)
        if seq_len > max_pos:
            # Use class-level warning counter to suppress repeated warnings
            if LLaDAModel._pos_warning_counter < LLaDAModel._pos_warning_max:
                print(f"Warning: Sequence length {seq_len} exceeds block size {self.config.block_size}. Using position embeddings padding.")
                LLaDAModel._pos_warning_counter += 1
                
                if LLaDAModel._pos_warning_counter == LLaDAModel._pos_warning_max:
                    print(f"Note: Suppressing further position embedding warnings. This is normal behavior.")
            
            # Use full sequence but pad position embeddings
            x = self.tok_emb(noisy_batch)
            
            # Get position embeddings for available positions
            pos_emb_available = self.pos_emb(pos)
            
            # Create padded position embeddings by repeating the last position embedding
            pos_emb = torch.zeros((1, seq_len, self.config.n_embd), device=device)
            pos_emb[:, :max_pos] = pos_emb_available
            
            # For positions beyond block_size, repeat the last position embedding
            if max_pos > 0:  # Safety check
                pos_emb[:, max_pos:] = pos_emb_available[:, -1:].expand(-1, seq_len - max_pos, -1)
            
            # Add position embeddings to token embeddings
            x = x + pos_emb
        else:
            # Normal processing when sequence fits within block size
            x = self.tok_emb(noisy_batch) + self.pos_emb(pos)
        
        x = self.drop(x)
        
        # Track router loss
        total_router_loss = torch.tensor(0.0, device=device)
        
        # Process through transformer blocks with error handling
        for i, block in enumerate(self.blocks):
            try:
                if self.use_checkpoint and self.training:
                    x, router_loss = checkpoint.checkpoint(block, x)
                else:
                    with torch.amp.autocast(enabled=True, device_type='cuda'):
                        x, router_loss = block(x)
                
                total_router_loss = total_router_loss + router_loss
            except Exception as e:
                print(f"Warning: Error in block {i}: {e}")
                # Skip this block on error and continue with the next one
                # This allows the model to potentially recover from errors
                continue
        
        x = self.ln_f(x)
        
        # Memory-efficient logits computation
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            # Ensure we're calculating logits for valid sequence length
            valid_x = x
            if seq_len < seq_len:
                valid_x = x[:, :seq_len]
            logits = self.lm_head(valid_x)
            
            # If we truncated the sequence, pad logits back to original size
            if seq_len < seq_len:
                pad_shape = list(logits.shape)
                pad_shape[1] = seq_len - seq_len
                padding = torch.zeros(pad_shape, dtype=logits.dtype, device=logits.device)
                logits = torch.cat([logits, padding], dim=1)
        
        # Free memory
        del x
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None and masked_indices is not None:
            try:
                # Safety checks for loss calculation
                if targets.shape != input_ids.shape:
                    print(f"Warning: targets shape {targets.shape} doesn't match input_ids shape {input_ids.shape}")
                    # Reshape targets to match input_ids if possible
                    if targets.numel() == input_ids.numel():
                        targets = targets.view(input_ids.shape)
                    else:
                        # Cannot calculate loss with mismatched shapes
                        return logits, None, total_router_loss
                
                # Calculate loss only on masked tokens
                flat_logits = logits.view(-1, logits.size(-1))
                flat_targets = targets.view(-1)
                flat_mask = masked_indices.view(-1)
                flat_p_mask = p_mask.view(-1)
                
                # Get indices of masked tokens
                masked_indices_flat = torch.where(flat_mask)[0]
                
                # Only calculate loss if there are masked tokens
                if masked_indices_flat.shape[0] > 0:
                    # Get logits and targets for masked tokens only
                    masked_logits = flat_logits[masked_indices_flat]
                    masked_targets = flat_targets[masked_indices_flat]
                    masked_p_mask = flat_p_mask[masked_indices_flat]
                    
                    # Calculate cross entropy loss
                    token_loss = F.cross_entropy(
                        masked_logits,
                        masked_targets,
                        reduction='none'
                    ) / masked_p_mask
                    
                    # Normalize loss
                    loss = token_loss.sum() / (batch_size * seq_len)
                    
                    # Clean up
                    del token_loss, masked_logits, masked_targets, masked_p_mask, masked_indices_flat
                else:
                    # No masked tokens to calculate loss on
                    loss = torch.tensor(0.0, device=device)
            except Exception as e:
                print(f"Warning: Error calculating loss: {e}")
                # Return zero loss in case of error
                loss = torch.tensor(0.0, device=device)
        
        return logits, loss, total_router_loss

    @torch.no_grad()
    def generate(self, prompt, steps=128, gen_length=128, block_length=128, temperature=None, remasking=None, tokenizer=None):
        """Generate text using LLaDA diffusion process"""
        try:
            device = prompt.device
            
            # Set temperature and remasking from config if not specified
            temperature = temperature if temperature is not None else self.config.temperature
            remasking = remasking if remasking is not None else self.config.remasking
            
            # Safety checks and clamping
            gen_length = max(1, min(gen_length, 1024))  # Limit max generation length
            block_length = max(1, min(block_length, gen_length))  # Ensure valid block length
            steps = max(1, min(steps, 100))  # Limit max steps
            
            # Make sure mask_token_id is safe
            safe_mask_token_id = min(self.mask_token_id, self.config.vocab_size - 1)
            
            # Initialize with fully masked sequence
            x = torch.full((1, prompt.shape[1] + gen_length), safe_mask_token_id, dtype=torch.long, device=device)
            # Safely copy prompt tokens
            prompt_len = min(prompt.shape[1], x.shape[1])
            x[:, :prompt_len] = prompt[:, :prompt_len].clone()
            
            # Clamp values to ensure all token IDs are within vocabulary
            x = torch.clamp(x, 0, self.config.vocab_size - 1)
            
            # Track prompt indices to avoid modifying them
            prompt_index = torch.zeros_like(x, dtype=torch.bool)
            prompt_index[:, :prompt_len] = True
            
            # Ensure block_length divides gen_length for simpler logic
            if gen_length % block_length != 0:
                block_length = gen_length  # Fallback to single block
            num_blocks = gen_length // block_length
            
            # Adjust steps per block
            steps_per_block = max(1, steps // max(1, num_blocks))
            
            # Generate each block
            for block_idx in range(num_blocks):
                # Get mask indices for current block
                start_idx = prompt_len + block_idx * block_length
                end_idx = min(x.shape[1], prompt_len + (block_idx + 1) * block_length)
                
                # Skip if this block is out of bounds
                if start_idx >= x.shape[1] or start_idx >= end_idx:
                    continue
                
                block_mask_index = (x[:, start_idx:end_idx] == safe_mask_token_id)
                
                # Skip if no tokens to unmask in this block
                mask_count = block_mask_index.sum(dim=1, keepdim=True)
                if mask_count.item() == 0:
                    continue
                
                # Calculate tokens to unmask at each step
                tokens_per_step = torch.div(mask_count, steps_per_block, rounding_mode='floor')
                tokens_per_step = torch.max(tokens_per_step, torch.ones_like(tokens_per_step))  # At least 1
                remainder = mask_count % steps_per_block
                
                # Use the scalar value from the tensor for torch.full
                tokens_per_step_scalar = tokens_per_step.item()
                num_transfer_tokens = torch.full((1, steps_per_block), tokens_per_step_scalar, device=device)
                if remainder > 0:
                    num_transfer_tokens[:, :remainder] += 1
                
                # Iteratively unmask tokens
                for step in range(steps_per_block):
                    # Check if we have any masked tokens left
                    if (x == safe_mask_token_id).sum() == 0:
                        break
                    
                    # Get current mask indices
                    mask_index = (x == safe_mask_token_id)
                    
                    # Get model predictions with explicit memory management
                    with torch.amp.autocast(enabled=True, device_type='cuda'):
                        # Safety check to ensure all token IDs are within vocabulary before forward pass
                        x_safe = torch.clamp(x, 0, self.config.vocab_size - 1)
                        logits, _, _ = self.forward(x_safe, apply_masking=False)
                    
                    # Apply optional temperature sampling
                    if temperature > 0:
                        # Add Gumbel noise for stochastic sampling
                        noise = torch.rand_like(logits)
                        gumbel_noise = -torch.log(-torch.log(noise + 1e-10) + 1e-10) * temperature
                        logits = logits + gumbel_noise
                    
                    # Get token predictions (ensuring they're within vocabulary range)
                    x0 = torch.argmax(logits, dim=-1)
                    x0 = torch.clamp(x0, 0, self.config.vocab_size - 1)
                    
                    # Calculate confidence scores for remasking strategy
                    if remasking == 'low_confidence':
                        p = F.softmax(logits, dim=-1)
                        # Safely gather values
                        x0_unsqueezed = x0.unsqueeze(-1)
                        if x0_unsqueezed.max() < p.shape[-1]:  # Safety check
                            x0_p = torch.gather(p, dim=-1, index=x0_unsqueezed).squeeze(-1)
                        else:
                            # Fallback if indices are out of range
                            print("Warning: Indices out of range in confidence calculation. Using random values.")
                            x0_p = torch.rand_like(x0, dtype=torch.float)
                        # Free memory
                        del p
                    else:  # Default to random remasking for any other value
                        x0_p = torch.rand_like(x0, dtype=torch.float)
                    
                    # Free memory for logits as they're no longer needed
                    del logits
                    torch.cuda.empty_cache()
                    
                    # Don't consider future blocks for unmasking
                    if end_idx < x.shape[1]:
                        x0_p[:, end_idx:] = float('-inf')
                    
                    # Don't modify prompt - use full_like for consistent device and dtype
                    neg_inf_tensor = torch.full_like(x0_p, float('-inf'))
                    x0_p = torch.where(prompt_index, neg_inf_tensor, x0_p)
                    
                    # Update predictions only for masked tokens
                    x0 = torch.where(mask_index, x0, x)
                    
                    # Use full_like for confidence calculation to ensure proper shape and device
                    neg_inf_tensor = torch.full_like(x0_p, float('-inf'))
                    confidence = torch.where(mask_index, x0_p, neg_inf_tensor)
                    
                    # Free memory
                    del x0_p, neg_inf_tensor
                    
                    # Select tokens to unmask based on confidence
                    transfer_index = torch.zeros_like(x, dtype=torch.bool)
                    for i in range(x.shape[0]):
                        # Count currently masked tokens
                        mask_count_current = mask_index[i].sum().item()
                        # Limit number of tokens to unmask to available masked tokens
                        k_transfer = min(int(num_transfer_tokens[i, step].item()), mask_count_current)
                        
                        if k_transfer > 0:
                            # Use topk to find indices of highest confidence predictions
                            if remasking == 'low_confidence':
                                if mask_count_current > 0:
                                    try:
                                        # Get only masked positions
                                        masked_pos = torch.where(mask_index[i])[0]
                                        if len(masked_pos) > 0:  # Safety check
                                            masked_conf = confidence[i, masked_pos]
                                            # Safety check for k value
                                            safe_k = min(k_transfer, len(masked_pos))
                                            if safe_k > 0:  # Ensure we have at least one element for topk
                                                # Get top-k from masked positions
                                                _, topk_local_idx = torch.topk(masked_conf, k=safe_k)
                                                # Map back to global indices
                                                if topk_local_idx.max() < len(masked_pos):  # Safety check
                                                    indices = masked_pos[topk_local_idx]
                                                    transfer_index[i, indices] = True
                                            # Free memory
                                            del masked_conf, topk_local_idx
                                        # Free memory
                                        del masked_pos
                                    except Exception as e:
                                        print(f"Warning: Error in confidence-based token selection: {e}")
                                        # Fallback to random selection
                                        masked_positions = torch.where(mask_index[i])[0]
                                        if len(masked_positions) > 0:
                                            indices = masked_positions[torch.randperm(len(masked_positions), device=device)[:k_transfer]]
                                            transfer_index[i, indices] = True
                            else:  # random remasking
                                # Randomly select indices among masked tokens
                                masked_positions = torch.where(mask_index[i])[0]
                                if len(masked_positions) > 0:
                                    perm = torch.randperm(len(masked_positions), device=device)
                                    # Safety check to ensure we don't go out of bounds
                                    safe_k = min(k_transfer, len(masked_positions))
                                    if safe_k > 0:
                                        indices = masked_positions[perm[:safe_k]]
                                        transfer_index[i, indices] = True
                                    # Free memory
                                    del perm
                                # Free memory
                                if 'masked_positions' in locals():
                                    del masked_positions
                    
                    # Update x with unmasked tokens
                    x = torch.where(transfer_index, x0, x)
                    
                    # Free memory for intermediate tensors
                    del x0, confidence, transfer_index, mask_index
                    torch.cuda.empty_cache()
                
                # Clear block variables to free memory
                del block_mask_index, num_transfer_tokens
                torch.cuda.empty_cache()
            
            # Detokenize the output if requested (and return both tokens and text)
            text_output = self.detokenize_output(x[0], tokenizer)
            
            return x, text_output
            
        except Exception as e:
            # Clear CUDA cache in case of error
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gc.collect()
            
            # Return error message as a list of integers for debugging
            error_message = f"Generation error: {str(e)}"
            error_tokens = [ord(c) for c in error_message]
            return torch.tensor([error_tokens], dtype=torch.long, device=prompt.device if hasattr(prompt, 'device') else 'cpu'), error_message

    @torch.no_grad()
    def get_log_likelihood(self, prompt, answer, mc_num=128, batch_size=16):
        """Estimate log-likelihood of answer given prompt using Monte Carlo"""
        device = prompt.device
        seq = torch.cat([prompt, answer])[None, :].repeat(batch_size, 1).to(device)
        prompt_index = torch.arange(seq.shape[1], device=device) < len(prompt)
        
        losses = []
        for _ in range(mc_num // batch_size):
            # Apply random masking with uniform sampling of mask count
            l = seq.shape[1] - len(prompt)  # answer length
            k = torch.randint(1, l + 1, (batch_size,), device=device)  # number of masks per batch
            
            mask_index = torch.zeros_like(seq, dtype=torch.bool)
            for i in range(batch_size):
                # Randomly select k[i] positions in the answer to mask
                answer_indices = torch.arange(len(prompt), seq.shape[1], device=device)
                perm = torch.randperm(l, device=device)
                to_mask = answer_indices[perm[:k[i]]]
                mask_index[i, to_mask] = True
            
            # Apply masking
            x = seq.clone()
            x[mask_index] = self.mask_token_id
            
            # Get model predictions
            logits, _, _ = self.forward(x, apply_masking=False)
            
            # Calculate loss only on masked tokens
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1))[mask_index.view(-1)],
                seq.view(-1)[mask_index.view(-1)],
                reduction='none'
            )
            
            # Adjust by mask ratio k/l
            mask_ratio = k.float() / l
            adjusted_loss = loss.sum() / mask_ratio.sum()
            
            losses.append(adjusted_loss.item())
        
        # Return negative log-likelihood
        return -sum(losses) / len(losses)

    def supervised_fine_tuning_step(self, input_ids, prompt_lengths, eps=1e-3):
        """Specialized forward pass for supervised fine-tuning"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Ensure all input IDs are within vocabulary range
        if torch.any(input_ids >= self.config.vocab_size):
            max_idx = torch.max(input_ids).item()
            print(f"Warning: Input contains token IDs outside vocabulary range in fine-tuning. Max ID: {max_idx}, Vocab size: {self.config.vocab_size}")
            input_ids = torch.clamp(input_ids, 0, self.config.vocab_size - 1)
        
        # Apply forward process (masking) to entire sequence
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            noisy_batch, masked_indices, p_mask = self.forward_process(input_ids, eps)
            
            # Do not add noise to the prompt part - keep it clean
            for i in range(batch_size):
                prompt_mask = torch.arange(seq_len, device=device) < prompt_lengths[i]
                noisy_batch[i, prompt_mask] = input_ids[i, prompt_mask]
            
            # Calculate the answer length (including padded tokens)
            answer_lengths = seq_len - prompt_lengths
            
            # Get model predictions
            logits, _, router_loss = self.forward(noisy_batch, apply_masking=False)
            
            # Only calculate loss on masked tokens in the response
            response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for i in range(batch_size):
                response_mask[i, prompt_lengths[i]:] = True
            
            masked_response = masked_indices & response_mask
            
            # Compute loss on masked response tokens only
            token_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1))[masked_response.view(-1)],
                input_ids.view(-1)[masked_response.view(-1)],
                reduction='none'
            ) / p_mask.view(-1)[masked_response.view(-1)]
            
            # Normalize by answer length
            ce_loss = 0
            for i in range(batch_size):
                i_loss = token_loss[i * seq_len + prompt_lengths[i]:(i+1) * seq_len]
                if len(i_loss) > 0:
                    ce_loss += i_loss.sum() / answer_lengths[i]
            
            ce_loss = ce_loss / batch_size
        
        return ce_loss, router_loss

    def set_gradient_checkpointing(self, value):
        """Enable or disable gradient checkpointing"""
        for block in self.blocks:
            block.use_checkpoint = value

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Configure optimizer with parameter groups for experts and routers"""
        # Collect all parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Create parameter groups
        expert_params = []
        router_params = []
        other_params = []
        
        for name, param in param_dict.items():
            if any(expert_type in name.lower() for expert_type in ['expert_group', 'shared_mlp', 'expert_proj', 'expert_adapters']):
                expert_params.append(param)
            elif 'router' in name.lower():
                router_params.append(param)
            else:
                other_params.append(param)
        
        # Create optimizer groups
        optim_groups = [
            # Expert parameters
            {
                'params': expert_params,
                'weight_decay': weight_decay,
                'lr': learning_rate
            },
            # Router parameters
            {
                'params': router_params,
                'weight_decay': 0.0,  # No weight decay for router
                'lr': learning_rate  # Lower learning rate for stability
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
        
        print(f"\nParameter counts:")
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
        
        return optimizer

    def detokenize_output(self, token_ids, tokenizer=None):
        """
        Convert token IDs back to text
        
        Args:
            token_ids: Tensor of token IDs
            tokenizer: Optional tokenizer instance. If None, will attempt to use a default tokenizer.
            
        Returns:
            Detokenized text
        """
        try:
            # If a tokenizer is provided, use it
            if tokenizer is not None:
                return tokenizer.decode(token_ids.tolist())
            
            # Try to import transformers and use a default tokenizer
            try:
                from transformers import AutoTokenizer
                default_tokenizer = AutoTokenizer.from_pretrained("gpt2")
                return default_tokenizer.decode(token_ids.tolist())
            except ImportError:
                # Fallback to a simple mapping if transformers is not available
                return f"[Tokenizer not available. Raw IDs: {token_ids.tolist()}]"
        except Exception as e:
            return f"Detokenization error: {str(e)}"

def generate_example_wrapper(model, prompt_tokens, tokenizer=None, steps=32, gen_length=32, temperature=0.7):
    """
    Wrapper function for generate_example that displays both token IDs and text.
    To be used in training scripts.
    
    Example usage in training script:
    ```
    # Replace:
    output_tokens = generate_example(model, prompt_tokens)
    print(f"Generated ids: {output_tokens.tolist()}")
    
    # With:
    from llada.llada import generate_example_wrapper
    generate_example_wrapper(model, prompt_tokens, tokenizer)
    ```
    """
    # Import HF tokenizer if needed and not provided
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            print("Using default GPT-2 tokenizer")
        except ImportError:
            print("Warning: transformers not installed. Text output will be unavailable.")
            tokenizer = None
    
    # Run generation with text output
    print("\nGenerating example...")
    
    try:
        with torch.no_grad():
            output_tokens, output_text = model.generate(
                prompt_tokens,
                steps=steps,
                gen_length=gen_length,
                temperature=temperature,
                tokenizer=tokenizer
            )
        
        # Display token IDs
        print(f"Generated ids: {output_tokens[0].tolist()}")
        
        # Display text
        print("\nGenerated text:")
        print("-" * 40)
        print(output_text)
        print("-" * 40)
        
        return output_tokens, output_text
        
    except Exception as e:
        print(f"Generation failed with error: {str(e)}")
        return None, str(e)

def generate_and_display(model, text=None, tokens=None, tokenizer=None, steps=32, gen_length=32, block_length=32, temperature=0.7, remasking='low_confidence'):
    """
    Convenience function to generate text and display the result in a readable format.
    
    Args:
        model: LLaDAModel instance
        text: Input text prompt. If None, tokens must be provided
        tokens: Input token IDs. If None, text must be provided
        tokenizer: Tokenizer to use. If None, will try to load a default one
        steps: Number of diffusion steps
        gen_length: Length of text to generate
        block_length: Block length for generation
        temperature: Temperature for generation
        remasking: Remasking strategy
        
    Returns:
        Tuple of (token_ids, generated_text)
    """
    import torch
    
    # Ensure we have either text or tokens
    if text is None and tokens is None:
        raise ValueError("Either text or tokens must be provided")
    
    # Get tokenizer if not provided
    if tokenizer is None and text is not None:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            print("Using default GPT-2 tokenizer")
        except ImportError:
            print("Transformers library not found, please provide tokens directly")
            return None, "Error: Tokenizer not available"
    
    # Get device
    device = next(model.parameters()).device
    
    # Convert text to tokens if needed
    if tokens is None and text is not None:
        tokens = tokenizer.encode(text, return_tensors="pt").to(device)
    elif tokens is not None and not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor([tokens], dtype=torch.long).to(device)
    elif tokens is not None and isinstance(tokens, torch.Tensor) and tokens.dim() == 1:
        tokens = tokens.unsqueeze(0).to(device)
    
    # Run generation
    try:
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            output_tokens, output_text = model.generate(
                tokens, 
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                remasking=remasking,
                tokenizer=tokenizer
            )
        
        # Format the output for display
        print("\n" + "="*50)
        print("GENERATED OUTPUT:")
        print("="*50)
        print(f"Token IDs: {output_tokens[0].tolist()}")
        print("-"*50)
        print("Generated text:")
        print("-"*50)
        print(output_text)
        print("="*50)
        
        return output_tokens, output_text
        
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return None, f"Error: {str(e)}"
