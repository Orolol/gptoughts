"""Multi-Head Latent Attention with Selective Attention - Fast Implementation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .positional_encoding import RoPE


class MLASelectiveFast(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer with Selective Attention - Optimized Version.
    
    This implementation uses fused operations and minimizes memory allocations
    for better GPU utilization.
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
        
        # Selective attention: reuse one attention head output for masking (no extra parameters)
        self.selection_head_idx = getattr(config, 'selection_head_idx', 0)
        
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
        
        # Set up caching for inference
        self.max_batch_size = getattr(config, 'max_batch_size', 8)
        self.max_seq_len = getattr(config, 'max_seq_len', 4096)
        self.attn_impl = getattr(config, 'attn_impl', "naive")
        
        # Initialize RoPE
        self.rope = RoPE(self.qk_rope_head_dim, self.max_seq_len)
        
        # Inference mode flag
        self.inference_mode = False
        
        # Pre-allocate buffers for selective attention to avoid reallocation
        self.register_buffer('_eye_mask', None, persistent=False)
        self.register_buffer('_causal_mask', None, persistent=False)
    
    def set_inference_mode(self, mode=True):
        """Set the module to inference mode (with caching) or training mode (no caching)."""
        if mode == self.inference_mode:
            return
            
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
        
        # If this is an inference context, initialize the caches
        if not self.training:
            self.set_inference_mode(True)
    
    def _get_or_create_masks(self, seqlen, device, dtype):
        """Get or create reusable masks for efficiency."""
        # Eye mask for diagonal zeroing
        if self._eye_mask is None or self._eye_mask.shape[0] < seqlen:
            self._eye_mask = torch.eye(seqlen, device=device, dtype=dtype)
        
        # Causal mask
        if self._causal_mask is None or self._causal_mask.shape[0] < seqlen:
            self._causal_mask = torch.triu(
                torch.full((seqlen, seqlen), float('-inf'), device=device, dtype=dtype),
                diagonal=1
            )
        
        return self._eye_mask[:seqlen, :seqlen], self._causal_mask[:seqlen, :seqlen]
    
    def _compute_selective_scores_fused(self, q_select, k_select, scale, eye_mask):
        """Fused computation of selective attention scores."""
        # Compute attention scores
        S = torch.matmul(q_select, k_select.transpose(-2, -1)) * scale
        S = S.squeeze(1)
        
        # Apply constraints in one fused operation
        S = F.relu(S)  # Remove inplace=True for torch.compile compatibility
        
        # Create mask to preserve BOS tokens
        bos_mask = torch.ones_like(S)
        bos_mask[:, :, 0] = 0
        S = S * bos_mask
        
        # Zero diagonal
        S = S * (1 - eye_mask)
        
        # Cumulative sum for selection mask
        return torch.cumsum(S, dim=-2)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        """Forward pass with optimized selective attention."""
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        
        # Apply query projections
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        
        # Reshape and split queries
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # Apply RoPE to q_pe
        q_pe = self.rope(q_pe.contiguous(), start_pos)
        
        # Process keys and values through low-rank projections
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        
        # Apply RoPE to k_pe
        k_pe_reshaped = k_pe.view(bsz, seqlen, 1, -1).transpose(1, 2)
        k_pe_encoded = self.rope(k_pe_reshaped.contiguous(), start_pos).squeeze(1)
        
        # Standard approach: compute full attention matrices
        q = torch.cat([q_nope, q_pe], dim=-1)
        kv = self.wkv_b(self.kv_norm(kv))
        kv = kv.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe_encoded.unsqueeze(2).expand(-1, -1, self.n_heads, -1)], dim=-1)
        
        # Check if we're in training or inference mode
        is_inference = self.inference_mode
        
        if is_inference and hasattr(self, 'k_cache') and hasattr(self, 'v_cache'):
            # Only update caches in inference mode
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            k_to_use = self.k_cache[:bsz, :end_pos]
            v_to_use = self.v_cache[:bsz, :end_pos]
        else:
            k_to_use = k
            v_to_use = v
        
        # Prepare tensors for SDPA
        q_sdpa = q.transpose(1, 2)  # [B, H, S, D]
        k_sdpa = k_to_use.transpose(1, 2)  # [B, H, T, D]
        v_sdpa = v_to_use.transpose(1, 2)  # [B, H, T, D]
        
        # Get reusable masks
        eye_mask, causal_mask = self._get_or_create_masks(seqlen, x.device, q_sdpa.dtype)
        
        # Extract selection head efficiently
        q_select = q_sdpa[:, self.selection_head_idx:self.selection_head_idx+1]
        k_select = k_sdpa[:, self.selection_head_idx:self.selection_head_idx+1]
        
        # Compute selective mask using JIT-compiled function
        selective_mask = self._compute_selective_scores_fused(
            q_select, k_select, self.softmax_scale, eye_mask
        ).unsqueeze(1)  # [B, 1, S, T]
        
        # Combine masks efficiently
        if mask is not None:
            attn_mask = mask.unsqueeze(0).unsqueeze(0) - selective_mask
        else:
            # For training, combine causal and selective masks
            if seqlen > 1:
                attn_mask = causal_mask.unsqueeze(0).unsqueeze(0) - selective_mask
            else:
                attn_mask = -selective_mask
        
        # Use SDPA with combined mask
        # Use the more efficient backend selection
        attn_output = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # We handle causality in the mask
            scale=self.softmax_scale
        )
        
        # Transpose and reshape
        x = attn_output.transpose(1, 2).contiguous()
        x = x.reshape(bsz, seqlen, -1)
        x = self.wo(x)
        
        # Handle FP8 conversion if needed
        if x.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            x = x.to(torch.bfloat16)
        
        x = self.resid_dropout(x)
        
        return x