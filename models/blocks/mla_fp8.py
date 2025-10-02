"""
MLA (Multi-Head Latent Attention) with FP8 optimizations based on DeepSeek-V3.

This module implements MLA with fine-grained FP8 quantization, high-precision
accumulation, and mixed precision strategies.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .positional_encoding import RoPE
from optimization.fp8_mla import FP8Quantizer, FP8LinearMLA


class MLA_FP8(nn.Module):
    """
    Multi-Head Latent Attention with FP8 optimizations following DeepSeek-V3 approach.
    
    Key features:
    - Fine-grained tile-wise quantization for activations
    - Block-wise quantization for weights
    - High-precision accumulation for FP8 operations
    - Mixed precision strategy keeping critical components in higher precision
    - Low-precision caching for inference
    """
    
    def __init__(self, config):
        super().__init__()
        self.dim = config.n_embd if hasattr(config, 'n_embd') else config.dim
        self.n_heads = config.n_head if hasattr(config, 'n_head') else config.n_heads
        
        # FP8 configuration
        self.use_fp8 = getattr(config, 'use_fp8', False)
        self.fp8_tile_size = getattr(config, 'fp8_tile_size', 128)
        self.fp8_mla_params = getattr(config, 'fp8_mla_params', False)
        
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
        
        # Create quantizer for FP8 operations
        if self.use_fp8:
            self.quantizer = FP8Quantizer(tile_size=self.fp8_tile_size)
        
        # Linear projections with FP8 support
        bias = config.bias if hasattr(config, 'bias') else False
        
        if self.use_fp8 and self.fp8_mla_params:
            # Use FP8-optimized linear layers
            if self.q_lora_rank == 0:
                self.wq = FP8LinearMLA(self.dim, self.n_heads * self.qk_head_dim, bias=bias)
            else:
                self.wq_a = FP8LinearMLA(self.dim, self.q_lora_rank, bias=bias)
                self.q_norm = nn.LayerNorm(self.q_lora_rank)  # Keep in high precision
                self.wq_b = FP8LinearMLA(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=bias)
            
            # Low-rank projection for keys and values
            self.wkv_a = FP8LinearMLA(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=bias)
            self.kv_norm = nn.LayerNorm(self.kv_lora_rank)  # Keep in high precision
            self.wkv_b = FP8LinearMLA(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=bias)
            
            # Output projection
            self.wo = FP8LinearMLA(self.n_heads * self.v_head_dim, self.dim, bias=bias)
        else:
            # Standard linear layers (will use FP8 during computation if enabled)
            if self.q_lora_rank == 0:
                self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=bias)
            else:
                self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=bias)
                self.q_norm = nn.LayerNorm(self.q_lora_rank)
                self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=bias)
            
            self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=bias)
            self.kv_norm = nn.LayerNorm(self.kv_lora_rank)
            self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=bias)
            
            self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=bias)
        
        # Dropout layers (kept in high precision)
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
        
        # Set up caching for inference
        self.max_batch_size = getattr(config, 'max_batch_size', 8)
        self.max_seq_len = getattr(config, 'max_seq_len', 4096)
        self.attn_impl = getattr(config, 'attn_impl', "absorb")
        
        # Initialize RoPE (kept in high precision)
        self.rope = RoPE(self.qk_rope_head_dim, self.max_seq_len)
        
        # Inference mode and cache initialization
        self.inference_mode = False
        self.k_cache = None
        self.v_cache = None
        self.kv_cache = None
        self.pe_cache = None
        
        # For low-precision caching (following DeepSeek)
        self.cache_dtype = torch.float8_e4m3fn if self.use_fp8 else torch.bfloat16
        self.use_low_precision_cache = getattr(config, 'fp8_cache', True)
        
    def _quantize_for_compute(self, x: torch.Tensor, is_activation: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Quantize tensor for FP8 computation if enabled.
        
        Args:
            x: Input tensor
            is_activation: Whether this is an activation (True) or weight (False)
            
        Returns:
            Tuple of (possibly quantized tensor, scale factors or None)
        """
        if not self.use_fp8 or not self.training:
            return x, None
        
        if is_activation:
            return self.quantizer.quantize_activation_tile(x)
        else:
            # For online weight quantization during forward pass
            # In practice, weights might be pre-quantized
            return x, None
    
    def _compute_attention_fp8(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                              mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attention with FP8 optimizations.
        
        Following DeepSeek:
        - Attention computation stays in higher precision
        - Input/output can be in FP8 for memory efficiency
        """
        # For attention, DeepSeek keeps computations in higher precision
        # But allows FP8 storage of activations
        
        # Ensure attention computation is in at least BF16
        compute_dtype = torch.bfloat16 if q.dtype in [torch.float8_e4m3fn, torch.float8_e5m2] else q.dtype
        
        if q.dtype != compute_dtype:
            q = q.to(compute_dtype)
            k = k.to(compute_dtype)
            v = v.to(compute_dtype)
        
        # Standard scaled dot-product attention
        # Using PyTorch's optimized implementation
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=mask is None and q.size(2) > 1,
            scale=self.softmax_scale
        )
        
        return attn_output
    
    def set_inference_mode(self, mode=True):
        """Set the module to inference mode with low-precision caching."""
        if mode == self.inference_mode:
            return
            
        self.inference_mode = mode
        
        if mode:
            # Create caches with appropriate precision
            cache_dtype = self.cache_dtype if self.use_low_precision_cache else torch.bfloat16
            
            if self.attn_impl == "naive":
                if not hasattr(self, "k_cache") or self.k_cache is None:
                    self.register_buffer("k_cache", torch.zeros(
                        self.max_batch_size, self.max_seq_len, self.n_local_heads, self.qk_head_dim,
                        dtype=cache_dtype
                    ), persistent=False)
                if not hasattr(self, "v_cache") or self.v_cache is None:
                    self.register_buffer("v_cache", torch.zeros(
                        self.max_batch_size, self.max_seq_len, self.n_local_heads, self.v_head_dim,
                        dtype=cache_dtype
                    ), persistent=False)
            else:
                # For low-rank caching
                if not hasattr(self, "kv_cache") or self.kv_cache is None:
                    self.register_buffer("kv_cache", torch.zeros(
                        self.max_batch_size, self.max_seq_len, self.kv_lora_rank,
                        dtype=cache_dtype
                    ), persistent=False)
                if not hasattr(self, "pe_cache") or self.pe_cache is None:
                    # Special E5M6 format for attention inputs (as mentioned by DeepSeek)
                    # For now, use standard cache dtype
                    self.register_buffer("pe_cache", torch.zeros(
                        self.max_batch_size, self.max_seq_len, self.qk_rope_head_dim,
                        dtype=cache_dtype
                    ), persistent=False)
        else:
            # Remove caches in training mode
            if hasattr(self, "k_cache"):
                delattr(self, "k_cache")
            if hasattr(self, "v_cache"):
                delattr(self, "v_cache")
            if hasattr(self, "kv_cache"):
                delattr(self, "kv_cache")
            if hasattr(self, "pe_cache"):
                delattr(self, "pe_cache")
    
    def forward(self, x: torch.Tensor, start_pos: int, 
                freqs_cis: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with FP8 optimizations.
        
        Mixed precision strategy:
        - Linear operations can use FP8
        - Normalization stays in high precision
        - Attention computation in at least BF16
        - Caching can use low precision for memory efficiency
        """
        # Always use training mode during training to prevent cache memory leaks
        if self.training and self.inference_mode:
            self.set_inference_mode(False)
        
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        
        # Store input dtype for final conversion
        input_dtype = x.dtype
        
        # Apply query projections
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            # Low-rank projection with normalization in high precision
            q_low = self.wq_a(x)
            q_norm = self.q_norm(q_low)  # Always high precision
            q = self.wq_b(q_norm)
        
        # Reshape and split queries
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # Apply rotary positional encoding (kept in high precision)
        if freqs_cis is not None:
            q_pe = self.rope(q_pe.contiguous(), start_pos)
        else:
            q_pe = self.rope(q_pe.contiguous())
        
        # Process keys and values through low-rank projections
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        
        # Apply rotary positional encoding to k_pe
        if freqs_cis is not None:
            k_pe = k_pe.view(bsz, seqlen, 1, -1)
            k_pe = k_pe.transpose(1, 2)
            k_pe = self.rope(k_pe.contiguous(), start_pos).squeeze(1)
        else:
            k_pe = k_pe.view(bsz, seqlen, 1, -1)
            k_pe = k_pe.transpose(1, 2)
            k_pe = self.rope(k_pe.contiguous()).squeeze(1)
        
        # Select attention implementation
        attn_impl = getattr(self, 'attn_impl', "absorb")
        is_inference = self.inference_mode
        
        if attn_impl == "naive":
            # Standard approach with potential FP8 caching
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv_norm_out = self.kv_norm(kv)  # High precision normalization
            kv = self.wkv_b(kv_norm_out)
            kv = kv.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)
            
            if is_inference and hasattr(self, 'k_cache') and hasattr(self, 'v_cache'):
                # Update caches with potential low-precision storage
                if self.use_low_precision_cache and k.dtype != self.cache_dtype:
                    self.k_cache[:bsz, start_pos:end_pos] = k.to(self.cache_dtype)
                    self.v_cache[:bsz, start_pos:end_pos] = v.to(self.cache_dtype)
                else:
                    self.k_cache[:bsz, start_pos:end_pos] = k
                    self.v_cache[:bsz, start_pos:end_pos] = v
                    
                k_to_use = self.k_cache[:bsz, :end_pos]
                v_to_use = self.v_cache[:bsz, :end_pos]
            else:
                k_to_use = k
                v_to_use = v
            
            # Reshape for attention computation
            q_sdpa = q.transpose(1, 2)
            k_sdpa = k_to_use.transpose(1, 2)
            v_sdpa = v_to_use.transpose(1, 2)
            
            # Compute attention (potentially with FP8 optimizations)
            attn_mask = None
            if mask is not None:
                attn_mask = mask.unsqueeze(0).unsqueeze(0).expand(bsz, self.n_heads, -1, -1)
            
            x = self._compute_attention_fp8(q_sdpa, k_sdpa, v_sdpa, attn_mask)
            x = x.transpose(1, 2)
            
        else:
            # Optimized low-rank approach with FP8 support
            wkv_b = self.wkv_b.weight
            wkv_b = wkv_b.view(self.n_heads, -1, self.kv_lora_rank)
            
            # Normalization in high precision
            kv_norm_tensor = self.kv_norm(kv)
            
            if is_inference and hasattr(self, 'kv_cache') and hasattr(self, 'pe_cache'):
                # Low-precision caching
                if self.use_low_precision_cache:
                    self.kv_cache[:bsz, start_pos:end_pos] = kv_norm_tensor.to(self.cache_dtype)
                    self.pe_cache[:bsz, start_pos:end_pos] = k_pe.to(self.cache_dtype)
                else:
                    self.kv_cache[:bsz, start_pos:end_pos] = kv_norm_tensor
                    self.pe_cache[:bsz, start_pos:end_pos] = k_pe
                    
                kv_to_use = self.kv_cache[:bsz, :end_pos]
                pe_to_use = self.pe_cache[:bsz, :end_pos]
            else:
                kv_to_use = kv_norm_tensor
                pe_to_use = k_pe
            
            # Ensure proper dtype for computation
            if kv_to_use.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                kv_to_use = kv_to_use.to(torch.bfloat16)
            if pe_to_use.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                pe_to_use = pe_to_use.to(torch.bfloat16)
            
            # Extract values from low-rank representation
            v = torch.einsum("btc,hdc->bthd", kv_to_use, wkv_b[:, -self.v_head_dim:])
            
            # Prepare queries and keys
            q_full = torch.cat([q_nope, q_pe], dim=-1)
            q_sdpa = q_full.transpose(1, 2)
            
            k_nope_full = torch.einsum("btc,hdc->bthd", kv_to_use, wkv_b[:, :self.qk_nope_head_dim])
            pe_expanded = pe_to_use.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
            k_full = torch.cat([k_nope_full, pe_expanded], dim=-1)
            k_sdpa = k_full.transpose(1, 2)
            v_sdpa = v.transpose(1, 2)
            
            # Compute attention
            attn_mask = None
            if mask is not None:
                attn_mask = mask.unsqueeze(0).unsqueeze(0).expand(bsz, self.n_heads, -1, -1)
            
            x = self._compute_attention_fp8(q_sdpa, k_sdpa, v_sdpa, attn_mask)
            x = x.transpose(1, 2)
        
        # Reshape and project to output dimension
        x = x.reshape(bsz, seqlen, -1)
        x = self.wo(x)
        
        # Apply dropout
        x = self.resid_dropout(x)
        
        # Ensure output dtype matches input for residual connections
        if x.dtype != input_dtype:
            x = x.to(input_dtype)
        
        return x
    
    def clear_cache(self):
        """Clear all caches - useful for memory management."""
        self.k_cache = None
        self.v_cache = None
        self.kv_cache = None
        self.pe_cache = None
        if hasattr(self, '_cache_tensors'):
            del self._cache_tensors