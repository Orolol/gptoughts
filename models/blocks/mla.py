"""Multi-Head Latent Attention (MLA) mechanisms for transformer models."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .positional_encoding import RoPE


try:
    from flash_attn import flash_attn_func_2
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False

try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

# Attention backends available
ATTENTION_BACKENDS = {
    'flash_attn_2': FLASH_ATTENTION_AVAILABLE,
    'xformers': XFORMERS_AVAILABLE,
    'sdpa': hasattr(F, 'scaled_dot_product_attention'),
    'standard': True
}

def get_best_attention_backend():
    """Returns the best available attention backend."""
    if ATTENTION_BACKENDS['flash_attn_2']:
        return 'flash_attn_2'
    elif ATTENTION_BACKENDS['xformers']:
        return 'xformers'
    elif ATTENTION_BACKENDS['sdpa']:
        return 'sdpa'
    else:
        return 'standard'

class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.
    
    MLA uses a low-rank projection for compressing the key-value representations,
    reducing memory usage and computational complexity while maintaining model quality.
    
    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
        attention_backend (str): Backend used for attention computation.
    """
    def __init__(self, config):
        super().__init__()
        self.dim = config.n_embd if hasattr(config, 'n_embd') else config.dim
        self.n_heads = config.n_head if hasattr(config, 'n_head') else config.n_heads
        
        # For distributed training
        self.world_size = getattr(config, 'world_size', 1)
        self.n_local_heads = self.n_heads // self.world_size
        
        # Low-rank dimensions
        self.q_lora_rank = getattr(config, 'q_lora_rank', 0)
        self.kv_lora_rank = getattr(config, 'kv_lora_rank', 512)
        
        # Head dimensions
        self.qk_nope_head_dim = getattr(config, 'qk_nope_head_dim', 128)
        self.qk_rope_head_dim = getattr(config, 'qk_rope_head_dim', 64)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = getattr(config, 'v_head_dim', 128)
        
        # Optional values from config
        self.dropout = getattr(config, 'dropout', 0.0)
        
        # Linear projections
        if self.q_lora_rank == 0:
            # Direct projection for queries
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=config.bias if hasattr(config, 'bias') else False)
        else:
            # Low-rank projection for queries
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=config.bias if hasattr(config, 'bias') else False)
            self.q_norm = nn.LayerNorm(self.q_lora_rank)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=config.bias if hasattr(config, 'bias') else False)
        
        # Low-rank projection for keys and values
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=config.bias if hasattr(config, 'bias') else False)
        self.kv_norm = nn.LayerNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=config.bias if hasattr(config, 'bias') else False)
        
        # Output projection
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=config.bias if hasattr(config, 'bias') else False)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        
        # Attention scaling factor
        self.softmax_scale = self.qk_head_dim ** -0.5
        
        # For extended sequences
        rope_factor = getattr(config, 'rope_factor', 1.0)
        if rope_factor > 1.0 and hasattr(config, 'original_seq_len') and hasattr(config, 'max_seq_len'):
            if config.max_seq_len > config.original_seq_len:
                mscale = getattr(config, 'mscale', 1.0)
                mscale = 0.1 * mscale * math.log(rope_factor) + 1.0
                self.softmax_scale = self.softmax_scale * mscale * mscale
        
        # Attention backend setup
        if hasattr(config, 'attention_backend') and config.attention_backend is not None:
            if config.attention_backend not in ATTENTION_BACKENDS:
                raise ValueError(f"Attention backend {config.attention_backend} not available")
            if not ATTENTION_BACKENDS[config.attention_backend]:
                print(f"Warning: {config.attention_backend} not available, falling back to best available backend")
                self.attention_backend = get_best_attention_backend()
            else:
                self.attention_backend = config.attention_backend
        else:
            self.attention_backend = get_best_attention_backend()
            
        print(f"Using attention backend for MLA: {self.attention_backend}")
        
        # Set up caching for inference
        self.max_batch_size = getattr(config, 'max_batch_size', 8)
        self.max_seq_len = getattr(config, 'max_seq_len', 4096)
        self.attn_impl = getattr(config, 'attn_impl', "absorb")
        
                # Initialize RoPE before anything else
        self.rope = RoPE(self.qk_rope_head_dim, self.max_seq_len)
        
        # Only create caches for inference, not for training
        # This prevents memory leaks during training
        self.inference_mode = False
        
    def set_inference_mode(self, mode=True):
        """
        Set the module to inference mode (with caching) or training mode (no caching).
        
        Args:
            mode (bool): True for inference mode, False for training mode
        """
        if mode == self.inference_mode:
            return  # No change needed
            
        self.inference_mode = mode
        
        # Create caches for inference mode if they don't exist
        if mode:
            if self.attn_impl == "naive":
                if not hasattr(self, "k_cache") or self.k_cache is None:
                    self.register_buffer("k_cache", torch.zeros(
                        self.max_batch_size, self.max_seq_len, self.n_local_heads, self.qk_head_dim
                    ), persistent=False)
                if not hasattr(self, "v_cache") or self.v_cache is None:
                    self.register_buffer("v_cache", torch.zeros(
                        self.max_batch_size, self.max_seq_len, self.n_local_heads, self.v_head_dim
                    ), persistent=False)
            else:
                if not hasattr(self, "kv_cache") or self.kv_cache is None:
                    self.register_buffer("kv_cache", torch.zeros(
                        self.max_batch_size, self.max_seq_len, self.kv_lora_rank
                    ), persistent=False)
                if not hasattr(self, "pe_cache") or self.pe_cache is None:
                    self.register_buffer("pe_cache", torch.zeros(
                        self.max_batch_size, self.max_seq_len, self.qk_rope_head_dim
                    ), persistent=False)
        else:
            # Remove caches in training mode to free memory
            if hasattr(self, "k_cache"):
                delattr(self, "k_cache")
            if hasattr(self, "v_cache"):
                delattr(self, "v_cache")
            if hasattr(self, "kv_cache"):
                delattr(self, "kv_cache")
            if hasattr(self, "pe_cache"):
                delattr(self, "pe_cache")
        


        # No need to create caches here now that we have set_inference_mode
        # If this is an inference context, initialize the caches
        if not self.training:
            self.set_inference_mode(True)
    
    def _memory_efficient_attention(self, q, k, v, mask=None, is_causal=True):
        """
        Wrapper for memory-efficient attention with xformers.
        """
        try:
            # Ensure all tensors have the same dtype
            working_dtype = torch.float16 if q.device.type == 'cuda' else torch.float32
            q = q.to(working_dtype)
            k = k.to(working_dtype)
            v = v.to(working_dtype)
            
            # Prepare tensors with memory pinning for faster transfer
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            
            # Handle expansion for GQA if necessary
            if k.size(1) != q.size(1):  # If dimensions don't match
                # Expand k and v to match the number of q heads
                k = k.expand(-1, q.size(1), -1, -1)
                v = v.expand(-1, q.size(1), -1, -1)
            
            # Reshape for xformers [B, H, T, D] -> [B, T, H, D]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Use optimized mask for xformers
            if is_causal:
                attn_bias = xops.LowerTriangularMask()
            else:
                # Convert mask to attention bias for xformers if needed
                attn_bias = xops.AttentionBias.from_mask(mask) if mask is not None else None
            
            # Use KV caching for generation
            if hasattr(self, '_cached_k') and hasattr(self, '_cached_v'):
                k = torch.cat([self._cached_k, k], dim=1)
                v = torch.cat([self._cached_v, v], dim=1)
                # Ensure cached tensors have the same dtype
                k = k.to(working_dtype)
                v = v.to(working_dtype)
                self._cached_k = k
                self._cached_v = v
            
            y = xops.memory_efficient_attention(
                q, k, v,
                attn_bias=attn_bias,
                p=self.attn_dropout.p if self.training else 0.0,
                scale=self.softmax_scale,
                op=None  # Let xformers choose the best operator
            )
            
            # Reshape back [B, T, H, D] -> [B, H, T, D]
            return y.transpose(1, 2)
            
        except Exception as e:
            print(f"xformers memory efficient attention failed: {e}")
            print(f"Tensor dtypes - q: {q.dtype}, k: {k.dtype}, v: {v.dtype}")
            return None
    
    def _flash_attention(self, q, k, v, mask=None, is_causal=True):
        """
        Wrapper for Flash Attention 2 with optimizations.
        """
        try:
            # Convert inputs to bfloat16
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
            
            # Use KV caching for generation
            if hasattr(self, '_cached_k') and hasattr(self, '_cached_v'):
                k = torch.cat([self._cached_k, k], dim=2)  # dim=2 for Flash Attention [B, H, T, D]
                v = torch.cat([self._cached_v, v], dim=2)
                self._cached_k = k
                self._cached_v = v
            
            with torch.amp.autocast(enabled=True, device_type='cuda'):
                # Apply Flash Attention with optimizations
                output = flash_attn_func_2(
                    q, k, v,
                    causal=is_causal,
                    softmax_scale=self.softmax_scale
                )
            
            return output
            
        except Exception as e:
            print(f"Flash Attention 2 failed: {e}")
            print(f"Input dtypes - q: {q.dtype}, k: {k.dtype}, v: {v.dtype}")
            print(f"Input shapes - q: {q.shape}, k: {k.shape}, v: {v.shape}")
            return None
    
    def _sdpa_attention(self, q, k, v, mask=None, is_causal=True):
        """
        Wrapper for Scaled Dot Product Attention with optimizations.
        """
        try:
            # Use KV caching for generation
            if hasattr(self, '_cached_k') and hasattr(self, '_cached_v'):
                k = torch.cat([self._cached_k, k], dim=2)
                v = torch.cat([self._cached_v, v], dim=2)
                self._cached_k = k
                self._cached_v = v
            
            # Use SDPA with optimizations
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None if is_causal else mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
                scale=self.softmax_scale
            )
        except Exception as e:
            print(f"SDPA failed: {e}")
            return None
    
    def _standard_attention(self, q, k, v, mask=None, is_causal=True):
        """
        Standard implementation of attention with memory optimizations.
        """
        # Use KV caching for generation
        if hasattr(self, '_cached_k') and hasattr(self, '_cached_v'):
            k = torch.cat([self._cached_k, k], dim=2)
            v = torch.cat([self._cached_v, v], dim=2)
            self._cached_k = k
            self._cached_v = v
        
        # Calculate attention with memory optimizations
        att = torch.empty(q.shape[:-2] + (q.shape[-2], k.shape[-2]), 
                         dtype=q.dtype, device=q.device)
        att = torch.baddbmm(
            att, q, k.transpose(-2, -1),
            beta=0, alpha=self.softmax_scale
        )
        
        # Clipping for numerical stability
        att = torch.clamp(att, min=-1e4, max=1e4)
        
        # Apply causal mask if needed
        if is_causal and mask is not None:
            att = att + mask
        
        # Use float32 for softmax for more stability
        att = F.softmax(att.float(), dim=-1).to(q.dtype)
        att = self.attn_dropout(att)
        
        # Use torch.bmm for the final product
        return torch.bmm(att, v)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (Optional[torch.Tensor]): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        
        # Apply query projections (either direct or low-rank)
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        
        # Reshape and split queries
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # Apply rotary positional encoding to q_pe
        if freqs_cis is not None:
            # Ensure input tensors have correct shape [B, H, T, D]
            q_pe = self.rope(q_pe.contiguous(), start_pos)
        else:
            # For backward compatibility with models that don't use freqs_cis
            q_pe = self.rope(q_pe.contiguous())
        
        # Process keys and values through low-rank projections
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        
        # Apply rotary positional encoding to k_pe
        if freqs_cis is not None:
            # We need to reshape k_pe to match rope's expected input format
            k_pe = k_pe.view(bsz, seqlen, 1, -1)  # [B, T, 1, D]
            k_pe = k_pe.transpose(1, 2)  # [B, 1, T, D]
            k_pe = self.rope(k_pe.contiguous(), start_pos).squeeze(1)  # [B, T, D]
        else:
            # We need to reshape k_pe to match rope's expected input format
            k_pe = k_pe.view(bsz, seqlen, 1, -1)  # [B, T, 1, D]
            k_pe = k_pe.transpose(1, 2)  # [B, 1, T, D]
            k_pe = self.rope(k_pe.contiguous()).squeeze(1)  # [B, T, D]
        
        # Select attention implementation approach
        attn_impl = getattr(self, 'attn_impl', "absorb")
        
        # Check if we're in training or inference mode
        is_inference = self.inference_mode
        
        if attn_impl == "naive":
            # Standard approach: compute full attention matrices
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)
            
            if is_inference and hasattr(self, 'k_cache') and hasattr(self, 'v_cache'):
                # Only update caches in inference mode
                self.k_cache[:bsz, start_pos:end_pos] = k
                self.v_cache[:bsz, start_pos:end_pos] = v
                # Use the cached values for attention
                k_to_use = self.k_cache[:bsz, :end_pos]
                v_to_use = self.v_cache[:bsz, :end_pos]
            else:
                # In training mode, just use the current batch values (no caching)
                # This dramatically reduces memory usage
                k_to_use = k
                v_to_use = v
            
            # Compute attention scores
            scores = torch.einsum("bshd,bthd->bsht", q, k_to_use) * self.softmax_scale
        else:
            # Optimized approach: use low-rank decomposition
            # Extract weight for wkv_b
            wkv_b = self.wkv_b.weight
            wkv_b = wkv_b.view(self.n_heads, -1, self.kv_lora_rank)
            
            # Project q_nope to the low-rank space
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            
            # Prepare normalized kv and pe tensors
            kv_norm_tensor = self.kv_norm(kv)
            
            if is_inference and hasattr(self, 'kv_cache') and hasattr(self, 'pe_cache'):
                # Only update caches in inference mode
                self.kv_cache[:bsz, start_pos:end_pos] = kv_norm_tensor  
                self.pe_cache[:bsz, start_pos:end_pos] = k_pe
                
                # Use the cached values for attention
                kv_to_use = self.kv_cache[:bsz, :end_pos]
                pe_to_use = self.pe_cache[:bsz, :end_pos]
            else:
                # In training mode, just use the current batch values (no caching)
                # This dramatically reduces memory usage
                kv_to_use = kv_norm_tensor
                pe_to_use = k_pe
            
            # Compute attention scores through efficient decomposition
            scores = (torch.einsum("bshc,btc->bsht", q_nope, kv_to_use) +
                    torch.einsum("bshr,btr->bsht", q_pe, pe_to_use)) * self.softmax_scale
        
        # Apply mask if provided
        if mask is not None:
            scores += mask.unsqueeze(1)
        
        # Apply softmax and dropout
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        scores = self.attn_dropout(scores)
        
        # Compute weighted sum with values
        if attn_impl == "naive":
            if is_inference and hasattr(self, 'v_cache'):
                x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
            else:
                x = torch.einsum("bsht,bthd->bshd", scores, v_to_use)
        else:
            # Project through the low-rank space for values
            if is_inference and hasattr(self, 'kv_cache'):
                x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            else:
                x = torch.einsum("bsht,btc->bshc", scores, kv_to_use)
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        
        # Reshape and project to output dimension
        x = x.reshape(bsz, seqlen, -1)
        x = self.wo(x)
        x = self.resid_dropout(x)
        
        return x