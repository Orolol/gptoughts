"""
MLA-LLaDA: Multi-head Latent Attention with Diffusion-based Language Modeling

This model combines:
- MLA (Multi-head Latent Attention) for memory-efficient attention
- LLaDA (Large Language Diffusion with mAsking) for bidirectional generation
- DynamicTanh normalization for faster training
- FP8 quantization for memory and compute efficiency
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from ..blocks.mla_fp8 import MLA_FP8
from ..blocks.mla import MLA
from ..blocks.mlp import MLP
from ..blocks.normalization import DynamicTanh, RMSNorm
from ..config import GPTConfig


@dataclass
class MLALLaDAConfig(GPTConfig):
    """Configuration for MLA-LLaDA model"""
    # Model dimensions
    hidden_size: int = 768
    num_layers: int = 12
    vocab_size: int = 50304
    
    # MLA specific
    q_lora_rank: int = 0  # 0 means full rank
    kv_lora_rank: int = 64  # Compressed latent dimension
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    
    # LLaDA specific
    mask_token_id: int = 126336
    max_diffusion_steps: int = 50
    min_diffusion_steps: int = 10
    mask_ratio_min: float = 0.15
    mask_ratio_max: float = 0.85
    remasking_strategy: str = "low_confidence"  # or "random"
    
    # FP8 configuration
    use_fp8: bool = False
    fp8_format: str = "e4m3"
    fp8_amax_history_len: int = 1024
    fp8_amax_compute_algo: str = "most_recent"
    
    # DynamicTanh
    use_dyt: bool = True
    dyt_init_alpha: float = 0.5
    
    # Diffusion cache optimization
    cache_compression_ratio: float = 0.5
    adaptive_cache: bool = True
    
    # Training
    intermediate_size: int = 2048
    dropout: float = 0.1
    gradient_checkpointing: bool = True
    
    def __post_init__(self):
        # Ensure compatibility
        self.num_attention_heads = self.hidden_size // (self.qk_nope_head_dim + self.qk_rope_head_dim)
        # Set defaults from parent class
        if not hasattr(self, 'n_embd'):
            self.n_embd = self.hidden_size
        if not hasattr(self, 'n_layer'):
            self.n_layer = self.num_layers
        if not hasattr(self, 'block_size'):
            self.block_size = 2048
        
        # Ensure mask_token_id is within bounds
        if self.mask_token_id >= self.vocab_size:
            self.mask_token_id = self.vocab_size - 1
            print(f"Warning: mask_token_id adjusted to {self.mask_token_id} to fit vocab_size {self.vocab_size}")
        elif self.mask_token_id < 0:
            self.mask_token_id = self.vocab_size - 1
            print(f"Warning: negative mask_token_id adjusted to {self.mask_token_id}")


class DiffusionCache:
    """Efficient cache management for diffusion steps"""
    
    def __init__(self, config: MLALLaDAConfig):
        self.config = config
        self.cache_layers = {}
        
    def store_step_cache(self, layer_idx: int, step: int, compressed_kv: torch.Tensor, 
                        mask_indices: Optional[torch.Tensor] = None):
        """Store compressed KV cache for a diffusion step"""
        cache_key = f"layer_{layer_idx}_step_{step}"
        
        if self.config.adaptive_cache and mask_indices is not None:
            # Only cache important tokens
            importance_scores = self._calculate_importance(compressed_kv, mask_indices)
            top_k = int(compressed_kv.size(1) * self.config.cache_compression_ratio)
            _, indices = torch.topk(importance_scores, top_k, dim=1)
            
            compressed_cache = torch.gather(
                compressed_kv, 1, 
                indices.unsqueeze(-1).expand(-1, -1, compressed_kv.size(-1))
            )
            
            self.cache_layers[cache_key] = {
                'data': compressed_cache,
                'indices': indices,
                'original_shape': compressed_kv.shape
            }
        else:
            self.cache_layers[cache_key] = {'data': compressed_kv}
    
    def retrieve_step_cache(self, layer_idx: int, step: int) -> Optional[torch.Tensor]:
        """Retrieve and decompress cache for a step"""
        cache_key = f"layer_{layer_idx}_step_{step}"
        
        if cache_key not in self.cache_layers:
            return None
            
        cache_data = self.cache_layers[cache_key]
        
        if 'indices' in cache_data:
            # Reconstruct from compressed cache
            batch_size, _, latent_dim = cache_data['data'].shape
            original_shape = cache_data['original_shape']
            
            # Initialize with zeros and scatter compressed values
            full_cache = torch.zeros(original_shape, device=cache_data['data'].device)
            full_cache.scatter_(1, 
                cache_data['indices'].unsqueeze(-1).expand(-1, -1, latent_dim),
                cache_data['data']
            )
            return full_cache
        
        return cache_data['data']
    
    def _calculate_importance(self, kv_states: torch.Tensor, mask_indices: torch.Tensor) -> torch.Tensor:
        """Calculate token importance based on attention patterns"""
        # Simple L2 norm as importance metric
        importance = torch.norm(kv_states, dim=-1)
        
        # Boost importance for masked tokens
        if mask_indices is not None:
            importance[mask_indices] *= 2.0
            
        return importance
    
    def clear(self):
        """Clear all cached states"""
        self.cache_layers.clear()


class BiDirectionalMLA(nn.Module):
    """MLA adapted for bidirectional attention in diffusion process"""
    
    def __init__(self, config: MLALLaDAConfig):
        super().__init__()
        self.config = config
        
        # Choose FP8 or standard MLA
        if config.use_fp8:
            self.mla = MLA_FP8(config)
        else:
            self.mla = MLA(config)
            
        # Diffusion-aware decomposition layers
        self.num_diffusion_layers = min(config.max_diffusion_steps, 5)  # Limit for memory
        
        if config.kv_lora_rank > 0:
            self.k_decompress_layers = nn.ModuleList([
                nn.Linear(config.kv_lora_rank, config.hidden_size)
                for _ in range(self.num_diffusion_layers)
            ])
            self.v_decompress_layers = nn.ModuleList([
                nn.Linear(config.kv_lora_rank, config.hidden_size)
                for _ in range(self.num_diffusion_layers)
            ])
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                diffusion_step: int = 0,
                past_key_values: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with bidirectional attention
        
        Returns:
            - attended_states: Output of attention
            - compressed_kv: Compressed KV states for caching
        """
        # Remove causal mask for bidirectional attention
        if attention_mask is not None:
            # Convert causal mask to simple padding mask
            attention_mask = attention_mask.max(dim=-1, keepdim=True)[0]
        
        # Apply MLA with optional diffusion-aware decomposition
        if hasattr(self, 'k_decompress_layers') and diffusion_step < self.num_diffusion_layers:
            # Custom decomposition based on diffusion step
            output = self.mla(
                x=hidden_states,
                start_pos=0,  # No caching for bidirectional attention
                freqs_cis=None,  # RoPE handled internally
                mask=attention_mask
            )
            
            # Apply step-specific transformation
            if isinstance(output, tuple) and len(output) > 1:
                attended_states, cache = output
                if cache is not None and 'compressed_kv' in cache:
                    compressed_kv = cache['compressed_kv']
                else:
                    compressed_kv = None
            else:
                attended_states = output
                compressed_kv = None
        else:
            # Standard MLA forward
            output = self.mla(
                x=hidden_states,
                start_pos=0,  # No caching for bidirectional attention
                freqs_cis=None,  # RoPE handled internally
                mask=attention_mask
            )
            
            if isinstance(output, tuple):
                attended_states = output[0]
                compressed_kv = output[1] if len(output) > 1 else None
            else:
                attended_states = output
                compressed_kv = None
                
        return attended_states, compressed_kv


class MaskAwareMLA(nn.Module):
    """MLA with integrated masking for diffusion"""
    
    def __init__(self, config: MLALLaDAConfig):
        super().__init__()
        self.config = config
        self.mla = BiDirectionalMLA(config)
        
        # Mask token embedding
        self.mask_token_embed = nn.Parameter(torch.randn(config.hidden_size))
        
        # Mask predictor with optional FP8
        if config.use_fp8:
            from ..blocks.mla_fp8 import FP8LinearMLA
            self.mask_predictor = FP8LinearMLA(config.hidden_size, config.vocab_size)
        else:
            self.mask_predictor = nn.Linear(config.hidden_size, config.vocab_size)
            
    def forward(self,
                hidden_states: torch.Tensor,
                input_ids: torch.Tensor,
                mask_ratio: Optional[float] = None,
                diffusion_step: int = 0,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward with optional masking
        
        Returns:
            - output_states or predictions
            - mask_indices (if masking applied)
            - compressed_cache
        """
        batch_size, seq_len = input_ids.shape
        
        # Apply masking if specified
        if mask_ratio is not None and mask_ratio > 0:
            mask_indices = self._apply_dynamic_masking(input_ids, mask_ratio)
            
            # Replace masked positions with mask embedding
            mask_embed = self.mask_token_embed.unsqueeze(0).unsqueeze(0)
            hidden_states = hidden_states.clone()
            hidden_states[mask_indices] = mask_embed
            
            # Also replace input_ids at masked positions with mask_token_id for consistency
            input_ids = input_ids.clone()
            input_ids[mask_indices] = min(self.config.mask_token_id, self.config.vocab_size - 1)
        else:
            mask_indices = None
            
        # Apply bidirectional MLA
        attended_states, compressed_cache = self.mla(
            hidden_states,
            attention_mask=attention_mask,
            diffusion_step=diffusion_step
        )
        
        # Predict masked tokens if needed
        if mask_indices is not None:
            predictions = self.mask_predictor(attended_states)
            return predictions, mask_indices, compressed_cache
            
        return attended_states, None, compressed_cache
    
    def _apply_dynamic_masking(self, input_ids: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """Apply masking with awareness of token importance"""
        batch_size, seq_len = input_ids.shape
        
        # Random masking baseline
        mask_probs = torch.rand(batch_size, seq_len, device=input_ids.device)
        
        # Avoid masking special tokens (padding, etc.)
        special_tokens = input_ids < 10  # Adjust based on tokenizer
        mask_probs[special_tokens] = 1.0  # Never mask
        
        # Create mask
        mask_indices = mask_probs < mask_ratio
        
        return mask_indices
    
    def _apply_training_masking(self, input_ids: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """Apply masking for training - similar to _apply_dynamic_masking but used in training_forward"""
        return self._apply_dynamic_masking(input_ids, mask_ratio)


class MLALLaDABlock(nn.Module):
    """Transformer block combining MLA attention with LLaDA diffusion"""
    
    def __init__(self, config: MLALLaDAConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Normalization layers (DyT or RMSNorm)
        if config.use_dyt:
            self.attention_norm = DynamicTanh(config.hidden_size, alpha_init=config.dyt_init_alpha)
            self.ffn_norm = DynamicTanh(config.hidden_size, alpha_init=config.dyt_init_alpha)
        else:
            self.attention_norm = RMSNorm(config.hidden_size)
            self.ffn_norm = RMSNorm(config.hidden_size)
        
        # MLA attention with masking
        self.attention = MaskAwareMLA(config)
        
        # Feed-forward network
        self.feed_forward = MLP(config)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = False
        
    def forward(self,
                hidden_states: torch.Tensor,
                input_ids: torch.Tensor,
                mask_ratio: Optional[float] = None,
                diffusion_step: int = 0,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        # Disable gradient checkpointing for now due to recursion issue
        # if self.gradient_checkpointing and self.training:
        #     return self._forward_with_checkpointing(
        #         hidden_states, input_ids, mask_ratio, diffusion_step, attention_mask
        #     )
        
        # Pre-norm attention
        normed_hidden = self.attention_norm(hidden_states)
        attention_output, mask_indices, cache = self.attention(
            normed_hidden,
            input_ids,
            mask_ratio=mask_ratio,
            diffusion_step=diffusion_step,
            attention_mask=attention_mask
        )
        
        # For mask prediction, return early
        if mask_indices is not None:
            return attention_output, mask_indices, cache
            
        # Residual connection
        hidden_states = hidden_states + self.dropout(attention_output)
        
        # Pre-norm FFN
        normed_hidden = self.ffn_norm(hidden_states)
        ff_output = self.feed_forward(normed_hidden)
        hidden_states = hidden_states + self.dropout(ff_output)
        
        return hidden_states, None, cache
    
    def _forward_with_checkpointing(self, *args):
        """Forward with gradient checkpointing"""
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
            
        return torch.utils.checkpoint.checkpoint(
            create_custom_forward(self._forward_impl),
            *args
        )
    
    def _forward_impl(self, hidden_states, input_ids, mask_ratio, diffusion_step, attention_mask):
        """Implementation for checkpointing"""
        return self.forward(hidden_states, input_ids, mask_ratio, diffusion_step, attention_mask)


class DiffusionScheduler:
    """Adaptive scheduler for diffusion process"""
    
    def __init__(self, config: MLALLaDAConfig):
        self.config = config
        self.max_steps = config.max_diffusion_steps
        self.min_steps = config.min_diffusion_steps
        
    def get_num_steps(self, sequence_length: int, complexity_score: Optional[float] = None) -> int:
        """Determine optimal number of diffusion steps"""
        # Base calculation
        base_steps = max(
            self.min_steps,
            min(self.max_steps, sequence_length // 16)
        )
        
        # Adjust for complexity if provided
        if complexity_score is not None:
            complexity_multiplier = 1.0 + (complexity_score - 0.5) * 0.5
            adjusted_steps = int(base_steps * complexity_multiplier)
            return max(self.min_steps, min(self.max_steps, adjusted_steps))
            
        return base_steps
    
    def get_mask_schedule(self, num_steps: int) -> torch.Tensor:
        """Generate masking schedule with cosine annealing"""
        steps = torch.linspace(1.0, 0.0, num_steps + 1)[:-1]
        # Cosine schedule for smoother denoising
        return 0.5 * (1 + torch.cos(math.pi * steps))


class MLALLaDAModel(nn.Module):
    """Combined MLA-LLaDA model with all optimizations"""
    
    def __init__(self, config: MLALLaDAConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Note: Positional encoding (RoPE) is handled inside MLA blocks
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            MLALLaDABlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        
        # Final normalization
        if config.use_dyt:
            self.norm = DynamicTanh(config.hidden_size, alpha_init=config.dyt_init_alpha)
        else:
            self.norm = RMSNorm(config.hidden_size)
        
        # Output projection (can be tied with embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Diffusion components
        self.diffusion_scheduler = DiffusionScheduler(config)
        self.cache_manager = DiffusionCache(config)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Enable gradient checkpointing if configured
        if config.gradient_checkpointing:
            self.enable_gradient_checkpointing()
            
    def _init_weights(self, module):
        """Initialize weights with proper scaling"""
        if isinstance(module, nn.Linear):
            # Scale initialization by hidden size for better gradient flow
            if hasattr(module, 'in_features'):
                std = 0.02 * (self.config.hidden_size / 768) ** 0.5
            else:
                std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Smaller std for embeddings with large vocab
            std = 0.02 * (768 / self.config.hidden_size) ** 0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        for layer in self.layers:
            layer.gradient_checkpointing = True
            
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                is_training: bool = True,
                num_diffusion_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with training/generation modes
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding
            labels: Target labels for training
            is_training: Whether in training mode
            num_diffusion_steps: Number of denoising steps for generation
            
        Returns:
            Dictionary with loss and/or generated tokens
        """
        if is_training and labels is not None:
            return self.training_forward(input_ids, attention_mask, labels)
        else:
            return self.generation_forward(input_ids, attention_mask, num_diffusion_steps)
            
    def training_forward(self,
                        input_ids: torch.Tensor,
                        attention_mask: Optional[torch.Tensor],
                        labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Training with random masking and diffusion loss"""
        batch_size, seq_len = input_ids.shape
        
        # Sample masking ratio uniformly (avoid .item() for torch.compile compatibility)
        mask_ratio_tensor = torch.rand(1, device=input_ids.device)
        mask_ratio = self.config.mask_ratio_min + mask_ratio_tensor * (
            self.config.mask_ratio_max - self.config.mask_ratio_min
        )
        # For torch.compile compatibility, avoid converting to scalar
        mask_ratio_scalar = mask_ratio.squeeze()
        
        # Apply masking to input_ids first
        masked_input_ids = input_ids.clone()
        mask_indices = self._apply_training_masking(input_ids, mask_ratio_scalar)
        
        if mask_indices is not None and mask_indices.any():
            # Replace masked positions in input_ids with mask token
            masked_input_ids[mask_indices] = self.config.mask_token_id
        
        # Get embeddings from masked input
        hidden_states = self.embed_tokens(masked_input_ids)
        
        # Forward through all layers normally
        for layer_idx, layer in enumerate(self.layers):
            # Pass through layer without additional masking
            output, _, cache = layer(
                hidden_states,
                masked_input_ids,
                mask_ratio=None,  # Don't apply masking in layers
                diffusion_step=0,
                attention_mask=attention_mask
            )
            hidden_states = output
                
        # Final projection
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # Calculate loss on all positions but weight masked tokens more heavily
        # Pass mask_indices to focus learning on masked positions
        loss = self.compute_diffusion_loss(logits, labels, mask_indices=mask_indices)
        
        return {
            'loss': loss,
            'logits': logits,
            'mask_ratio': mask_ratio_scalar  # Keep as tensor for logging
        }
    
    def generation_forward(self,
                          input_ids: torch.Tensor,
                          attention_mask: Optional[torch.Tensor],
                          num_diffusion_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Generation with iterative denoising"""
        batch_size, seq_len = input_ids.shape
        
        # Detect prompt length (non-masked tokens)
        prompt_mask = input_ids != self.config.mask_token_id
        prompt_lengths = prompt_mask.sum(dim=1)
        
        # Initialize with masked response
        current_ids = input_ids.clone()
        for b in range(batch_size):
            current_ids[b, prompt_lengths[b]:] = self.config.mask_token_id
            
        # Determine number of diffusion steps
        if num_diffusion_steps is None:
            num_diffusion_steps = self.diffusion_scheduler.get_num_steps(seq_len)
            
        # Get masking schedule
        mask_schedule = self.diffusion_scheduler.get_mask_schedule(num_diffusion_steps)
        
        # Clear cache
        self.cache_manager.clear()
        
        # Iterative denoising
        for step in range(num_diffusion_steps):
            mask_ratio = mask_schedule[step].item()
            
            # Get embeddings
            hidden_states = self.embed_tokens(current_ids)
            
            # Forward through layers
            for layer_idx, layer in enumerate(self.layers):
                # Retrieve cached states if available
                cached_kv = self.cache_manager.retrieve_step_cache(layer_idx, step - 1)
                
                output, mask_indices, cache = layer(
                    hidden_states,
                    current_ids,
                    mask_ratio=mask_ratio if step < num_diffusion_steps - 1 else 0.0,
                    diffusion_step=step,
                    attention_mask=attention_mask
                )
                
                # Store cache for next step
                if cache is not None:
                    self.cache_manager.store_step_cache(layer_idx, step, cache)
                
                # Update hidden states only if not masked output
                if mask_indices is None:
                    hidden_states = output
                    
            # Final projection - check if we have predictions or need to project
            if mask_indices is not None:
                # Last layer returned predictions
                logits = output
            else:
                # Need to project hidden states
                hidden_states = self.norm(hidden_states)
                logits = self.lm_head(hidden_states)
            
            # Update tokens based on confidence
            if step < num_diffusion_steps - 1:
                current_ids = self.selective_remasking(
                    current_ids, logits, mask_ratio, prompt_lengths
                )
            else:
                # Final step: unmask all
                predicted_tokens = torch.argmax(logits, dim=-1)
                # Ensure predicted tokens are within bounds
                predicted_tokens = torch.clamp(predicted_tokens, 0, self.config.vocab_size - 1)
                for b in range(batch_size):
                    current_ids[b, prompt_lengths[b]:] = predicted_tokens[b, prompt_lengths[b]:]
                    
        return {
            'generated_ids': current_ids,
            'num_steps': num_diffusion_steps
        }
    
    def selective_remasking(self,
                           current_ids: torch.Tensor,
                           logits: torch.Tensor,
                           mask_ratio: float,
                           prompt_lengths: torch.Tensor) -> torch.Tensor:
        """Selectively remask tokens based on prediction confidence with repetition penalty"""
        batch_size, seq_len = current_ids.shape
        
        # Apply repetition penalty to logits
        logits = self._apply_repetition_penalty(logits, current_ids, penalty=1.2)
        
        # Calculate confidence scores
        probs = F.softmax(logits / 0.8, dim=-1)  # Temperature scaling for diversity
        confidence_scores = torch.max(probs, dim=-1)[0]
        
        # Create new IDs
        new_ids = current_ids.clone()
        
        # Sample from distribution instead of argmax to avoid repetition
        predicted_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch_size, seq_len)
        # Ensure predicted tokens are within bounds
        predicted_tokens = torch.clamp(predicted_tokens, 0, self.config.vocab_size - 1)
        
        for b in range(batch_size):
            response_start = prompt_lengths[b].item()
            response_confidence = confidence_scores[b, response_start:]
            
            if self.config.remasking_strategy == "low_confidence":
                # Find confidence threshold
                if mask_ratio > 0:
                    threshold = torch.quantile(response_confidence, q=mask_ratio)
                    
                    # Update tokens
                    high_conf_mask = response_confidence >= threshold
                    low_conf_mask = ~high_conf_mask
                    
                    # Unmask high confidence tokens
                    response_slice = slice(response_start, seq_len)
                    is_masked = new_ids[b, response_slice] == self.config.mask_token_id
                    
                    unmask_indices = is_masked & high_conf_mask
                    new_ids[b, response_start:][unmask_indices] = predicted_tokens[b, response_start:][unmask_indices]
                    
                    # Remask low confidence tokens
                    remask_indices = ~is_masked & low_conf_mask
                    new_ids[b, response_start:][remask_indices] = self.config.mask_token_id
            else:
                # Random remasking
                num_to_mask = int(len(response_confidence) * mask_ratio)
                if num_to_mask > 0:
                    indices = torch.randperm(len(response_confidence))[:num_to_mask]
                    new_ids[b, response_start + indices] = self.config.mask_token_id
                    
                # Unmask some random tokens
                is_masked = new_ids[b, response_start:] == self.config.mask_token_id
                num_to_unmask = min(int(is_masked.sum() * 0.3), is_masked.sum() - num_to_mask)
                if num_to_unmask > 0:
                    masked_positions = torch.where(is_masked)[0]
                    unmask_positions = masked_positions[torch.randperm(len(masked_positions))[:num_to_unmask]]
                    new_ids[b, response_start + unmask_positions] = predicted_tokens[b, response_start + unmask_positions]
                    
        return new_ids
    
    def _apply_repetition_penalty(self, logits: torch.Tensor, input_ids: torch.Tensor, penalty: float = 1.2) -> torch.Tensor:
        """Apply repetition penalty to reduce repetitive outputs"""
        if penalty == 1.0:
            return logits
            
        # Clone logits to avoid in-place modification
        logits = logits.clone()
        
        # Get unique tokens in the sequence
        for i in range(input_ids.size(0)):
            unique_ids = torch.unique(input_ids[i])
            # Apply penalty to tokens that have appeared
            for token_id in unique_ids:
                if token_id != self.config.mask_token_id and token_id >= 0:
                    logits[i, :, token_id] = logits[i, :, token_id] / penalty
                    
        return logits
    
    def _apply_training_masking(self, input_ids: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """Apply masking for training"""
        batch_size, seq_len = input_ids.shape
        
        # Random masking baseline
        mask_probs = torch.rand(batch_size, seq_len, device=input_ids.device)
        
        # Avoid masking special tokens (padding, etc.)
        special_tokens = input_ids < 10  # Adjust based on tokenizer
        mask_probs[special_tokens] = 1.0  # Never mask
        
        # Create mask
        mask_indices = mask_probs < mask_ratio
        
        return mask_indices
    
    def compute_diffusion_loss(self,
                              logits: torch.Tensor,
                              labels: torch.Tensor,
                              mask_indices: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute loss with improved stability and gradient flow"""
        # Clone labels for safe modification
        labels = labels.clone()
        
        # FIRST: Convert -1 tokens to -100 before any other processing
        # This fixes the core issue with invalid tokens from the data loader
        invalid_padding_mask = (labels == -1)
        if invalid_padding_mask.any():
            labels[invalid_padding_mask] = -100
        
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Get vocab size for validation
        vocab_size = shift_logits.size(-1)
        
        # Handle any remaining out-of-bounds tokens efficiently
        # Valid tokens: 0 <= token < vocab_size OR token == -100 (ignore index)
        out_of_bounds_high = shift_labels >= vocab_size
        out_of_bounds_low = (shift_labels < 0) & (shift_labels != -100)
        out_of_bounds = out_of_bounds_high | out_of_bounds_low
        
        if out_of_bounds.any():
            shift_labels[out_of_bounds] = -100
        
        # Don't ignore mask tokens during training - we want to predict them!
        # This is a key fix - the model needs to learn to predict mask tokens
        
        # Check for valid tokens
        non_ignore_mask = (shift_labels != -100)
        num_valid_tokens = non_ignore_mask.sum().item()
        
        if num_valid_tokens == 0:
            # Return a small loss to maintain gradient flow
            return torch.tensor(0.01, device=logits.device, dtype=logits.dtype, requires_grad=True)
        
        # Calculate cross-entropy loss without label smoothing initially
        # Label smoothing can make training harder initially
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100, label_smoothing=0.0)
        logits_flat = shift_logits.view(-1, shift_logits.size(-1))
        labels_flat = shift_labels.view(-1)
        
        # Compute loss
        loss = loss_fct(logits_flat, labels_flat)
        loss = loss.view(shift_labels.size())
        
        # For diffusion training, we compute loss on all valid tokens
        # This helps the model learn the overall structure better
        # Apply weighting based on whether tokens were masked
        if mask_indices is not None and mask_indices.any():
            shift_mask = mask_indices[..., 1:].contiguous()
            # Weight masked tokens more heavily (2x) to focus learning
            loss_weights = torch.ones_like(loss)
            loss_weights[shift_mask] = 2.0
            loss = loss * loss_weights
            
        # Average over valid tokens
        loss = loss[non_ignore_mask].mean()
        
        # Disable L2 regularization on logits for now - it can hurt initial training
        # logit_reg = 0.001 * (shift_logits ** 2).mean()
        # loss = loss + logit_reg
        
        return loss
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, optimizer_type=None, **kwargs):
        """Configure optimizer based on type"""
        # Get all parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Create parameter groups with weight decay
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # Select optimizer based on type
        if optimizer_type == 'lion':
            try:
                from lion_pytorch import Lion
                print("Using Lion optimizer")
                return Lion(optim_groups, lr=learning_rate, betas=betas)
            except ImportError:
                print("Lion optimizer not available, falling back to AdamW")
                optimizer_type = 'adamw'
                
        if optimizer_type == 'apollo':
            try:
                from apollo import Apollo
                print("Using Apollo optimizer")
                return Apollo(optim_groups, lr=learning_rate, weight_decouple=True)
            except ImportError:
                print("Apollo optimizer not available, falling back to AdamW")
                optimizer_type = 'adamw'
                
        # Default to AdamW
        print(f"Using AdamW optimizer (optimizer_type={optimizer_type})")
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    
    @torch.no_grad()
    def estimate_complexity(self, input_ids: torch.Tensor) -> float:
        """Estimate sequence complexity for adaptive diffusion steps"""
        # Simple heuristic based on token diversity
        unique_tokens = torch.unique(input_ids, dim=1).size(1)
        seq_len = input_ids.size(1)
        
        # Normalize to [0, 1]
        diversity_score = unique_tokens / seq_len
        
        # Adjust based on sequence length
        length_factor = min(1.0, seq_len / 512)
        
        return diversity_score * length_factor
    
    @torch.no_grad()
    def generate(self, 
                 prompt: Optional[torch.Tensor] = None,
                 input_ids: Optional[torch.Tensor] = None,
                 idx: Optional[torch.Tensor] = None,
                 gen_length: int = 50,
                 max_new_tokens: int = 50,
                 temperature: float = 0.8,
                 top_k: int = 40,
                 **kwargs) -> Tuple[torch.Tensor, None]:
        """Generate text using diffusion process
        
        Args:
            prompt: Input token IDs [batch_size, seq_len] (legacy)
            input_ids: Input token IDs [batch_size, seq_len] (preferred)
            idx: Input token IDs [batch_size, seq_len] (MLA compatibility)
            gen_length: Number of tokens to generate
            max_new_tokens: Number of tokens to generate (alternative name)
            temperature: Not used in diffusion generation
            top_k: Not used in diffusion generation
            
        Returns:
            Tuple of (generated_ids, None)
        """
        # Handle multiple parameter names for compatibility
        if input_ids is not None:
            prompt = input_ids
        elif idx is not None:
            prompt = idx
        elif prompt is None:
            raise ValueError("Either prompt, input_ids, or idx must be provided")
        
        # Use max_new_tokens if provided, otherwise use gen_length
        if max_new_tokens != 50:  # If max_new_tokens was explicitly set
            gen_length = max_new_tokens
            
        batch_size, prompt_len = prompt.shape
        
        # Create full sequence with mask tokens
        full_length = min(prompt_len + gen_length, self.config.block_size)
        input_ids = torch.full((batch_size, full_length), self.config.mask_token_id, 
                              dtype=prompt.dtype, device=prompt.device)
        input_ids[:, :prompt_len] = prompt
        
        # Run generation
        output = self.generation_forward(
            input_ids=input_ids,
            attention_mask=None,
            num_diffusion_steps=kwargs.get('num_diffusion_steps', None)
        )
        
        return output['generated_ids'], None
    
    def generate_simple(self, 
                       idx: torch.Tensor,
                       max_new_tokens: int = 50,
                       temperature: float = 0.8,
                       top_k: int = 40) -> torch.Tensor:
        """Simple autoregressive generation for debugging"""
        # Take the conditioning sequence, which will be used to predict the next token
        device = idx.device
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass to get logits
            with torch.no_grad():
                hidden_states = self.embed_tokens(idx_cond)
                for layer in self.layers:
                    output, _, _ = layer(
                        hidden_states,
                        idx_cond,
                        mask_ratio=None,
                        diffusion_step=0,
                        attention_mask=None
                    )
                    hidden_states = output
                hidden_states = self.norm(hidden_states)
                logits = self.lm_head(hidden_states)
            
            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


def create_mla_llada_model(config: Optional[MLALLaDAConfig] = None) -> MLALLaDAModel:
    """Factory function to create MLA-LLaDA model"""
    if config is None:
        config = MLALLaDAConfig()
    return MLALLaDAModel(config)