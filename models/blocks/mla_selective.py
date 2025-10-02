"""Multi-Head Latent Attention with Selective Attention mechanisms for transformer models."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .positional_encoding import RoPE
# Remove FlexAttention import as we'll use standard SDPA
# from torch.nn.attention.flex_attention import flex_attention, create_block_mask


class MLASelective(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer with Selective Attention.
    
    This implementation combines MLA's low-rank projections with selective attention
    mechanism from "Selective Attention Improves Transformer" (arXiv:2410.02703).
    
    The selective attention mechanism works as follows:
    1. Reuses the first attention head's logits as selection scores S
    2. Applies three constraints to S:
       - ReLU to ensure non-negative values
       - Zeros out first column to preserve BOS token
       - Zeros out diagonal to prevent self-masking
    3. Accumulates masks causally: F = cumsum(S, axis=-2)
    4. Subtracts F from attention logits before softmax
    
    Key features:
    - Parameter-free: reuses existing attention head computation
    - Memory efficient: 16-47x reduction in attention buffer size
    - Performance: equivalent to 2x larger standard attention modules
    
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
        selection_ratio (float): Ratio of tokens to select (0.0 to 1.0).
        selection_method (str): Method for token selection ('top_k', 'threshold', 'gumbel').
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
        
        # Selective attention is always enabled in MLASelective
        
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
        # Following the paper "Selective Attention Improves Transformer" (arXiv:2410.02703)
        self.selection_head_idx = getattr(config, 'selection_head_idx', 0)  # Which head to use for selection
        
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
    
    def _compute_selective_mask_efficient(self, q_sdpa, k_sdpa, seqlen, device):
        """
        Compute selective attention mask efficiently without FlexAttention.
        
        Args:
            q_sdpa: Query tensor [B, H, S, D]
            k_sdpa: Key tensor [B, H, T, D]
            seqlen: Sequence length
            device: Device
            
        Returns:
            selective_mask: Tensor to be subtracted from attention scores [B, H, S, T]
        """
        # Extract selection head and compute attention scores in one go
        # Use slicing which is more efficient than indexing
        q_select = q_sdpa[:, self.selection_head_idx:self.selection_head_idx+1, :, :]  # [B, 1, S, D]
        k_select = k_sdpa[:, self.selection_head_idx:self.selection_head_idx+1, :, :]  # [B, 1, T, D]
        
        # Fused attention computation with scaling
        S = torch.matmul(q_select, k_select.transpose(-2, -1)) * self.softmax_scale  # [B, 1, S, T]
        S = S.squeeze(1)  # [B, S, T]
        
        # Apply all constraints in-place for better memory efficiency
        S.relu_()  # Constraint 1: ReLU in-place
        S[:, :, 0] = 0  # Constraint 2: Preserve BOS token
        
        # Constraint 3: Zero diagonal efficiently
        # Use stride tricks to avoid creating a full diagonal mask
        diagonal_stride = S.stride(1) + S.stride(2)
        S.view(-1)[::diagonal_stride] = 0
        
        # Accumulate selection mask in-place if possible
        selective_mask = torch.cumsum(S, dim=-2)  # [B, S, T]
        
        # Broadcast to all heads without explicit expand (more memory efficient)
        selective_mask = selective_mask.unsqueeze(1)  # [B, 1, S, T]
        
        return selective_mask
    
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer with Selective Attention.

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
        
        # Selective attention will be applied after computing attention weights
        
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
            
            # Prepare tensors for SDPA
            q_sdpa = q.transpose(1, 2)  # [B, H, S, D]
            k_sdpa = k_to_use.transpose(1, 2)  # [B, H, T, D]
            v_sdpa = v_to_use.transpose(1, 2)  # [B, H, T, D]
            
            # Compute selective mask efficiently
            selective_mask = self._compute_selective_mask_efficient(q_sdpa, k_sdpa, seqlen, x.device)
            
            # Create causal mask and combine with selective mask
            if mask is not None:
                # Expand mask from [S, T] to [B, H, S, T]
                attn_mask = mask.unsqueeze(0).unsqueeze(0).expand(bsz, self.n_heads, -1, -1)
                # Subtract selective mask from attention scores
                attn_mask = attn_mask - selective_mask
            else:
                # Create causal mask efficiently
                with torch.no_grad():
                    causal_mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device, dtype=q_sdpa.dtype) * float('-inf'), diagonal=1)
                attn_mask = causal_mask.unsqueeze(0).unsqueeze(0) - selective_mask
            
            # Use standard SDPA with the combined mask
            attn_output = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,  # We handle causality in the mask
                scale=self.softmax_scale
            )
            
            # Transpose back to [B, S, H, D]
            x = attn_output.transpose(1, 2)
        else:
            # Optimized approach: use low-rank decomposition
            # Extract weight for wkv_b
            wkv_b = self.wkv_b.weight
            wkv_b = wkv_b.view(self.n_heads, -1, self.kv_lora_rank)
            
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
            
            # For the optimized approach, we need to project values through low-rank space
            # First, extract values from the low-rank representation
            v = torch.einsum("btc,hdc->bthd", kv_to_use, wkv_b[:, -self.v_head_dim:])
            
            # Reshape queries for SDPA
            q_full = torch.cat([q_nope, q_pe], dim=-1)  # Combine q components
            q_sdpa = q_full.transpose(1, 2)  # [B, H, S, D]
            
            # For keys, we need to reconstruct from low-rank space
            k_nope_full = torch.einsum("btc,hdc->bthd", kv_to_use, wkv_b[:, :self.qk_nope_head_dim])
            k_full = torch.cat([k_nope_full, pe_to_use.unsqueeze(2).expand(-1, -1, self.n_heads, -1)], dim=-1)
            k_sdpa = k_full.transpose(1, 2)  # [B, H, T, D]
            v_sdpa = v.transpose(1, 2)  # [B, H, T, D]
            
            # Compute selective mask efficiently
            selective_mask = self._compute_selective_mask_efficient(q_sdpa, k_sdpa, seqlen, x.device)
            
            # Create causal mask and combine with selective mask
            if mask is not None:
                # Expand mask from [S, T] to [B, H, S, T]
                attn_mask = mask.unsqueeze(0).unsqueeze(0).expand(bsz, self.n_heads, -1, -1)
                # Subtract selective mask from attention scores
                attn_mask = attn_mask - selective_mask
            else:
                # Create causal mask efficiently
                with torch.no_grad():
                    causal_mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device, dtype=q_sdpa.dtype) * float('-inf'), diagonal=1)
                attn_mask = causal_mask.unsqueeze(0).unsqueeze(0) - selective_mask
            
            # Use standard SDPA with the combined mask
            attn_output = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,  # We handle causality in the mask
                scale=self.softmax_scale
            )
            
            # Transpose back to [B, S, H, D]
            x = attn_output.transpose(1, 2)
        
        # Reshape and project to output dimension
        x = x.reshape(bsz, seqlen, -1)
        x = self.wo(x)
        
        # Handle FP8 conversion: ensure output matches the expected dtype for residual connections
        # Convert back to BFloat16 if the output is in FP8 format
        if x.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            x = x.to(torch.bfloat16)
        
        x = self.resid_dropout(x)
        
        return x