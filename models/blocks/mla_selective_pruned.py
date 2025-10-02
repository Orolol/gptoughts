"""Multi-Head Latent Attention with Selective Attention and Dynamic Pruning."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .positional_encoding import RoPE


class MLASelectivePruned(nn.Module):
    """
    MLA with Selective Attention and Dynamic Token Pruning.
    
    This implementation prunes tokens dynamically based on selection scores,
    reducing memory usage during training and inference.
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
        
        # Pruning parameters
        self.pruning_ratio = getattr(config, 'pruning_ratio', 0.5)  # Keep top 50% of tokens
        self.min_kept_tokens = getattr(config, 'min_kept_tokens', 128)  # Always keep at least 128 tokens
        
        # Optional values from config
        self.dropout = getattr(config, 'dropout', 0.0)
        
        # Linear projections
        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=config.bias if hasattr(config, 'bias') else False)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=config.bias if hasattr(config, 'bias') else False)
            self.q_norm = nn.LayerNorm(self.q_lora_rank)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=config.bias if hasattr(config, 'bias') else False)
        
        # Low-rank projection for keys and values
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=config.bias if hasattr(config, 'bias') else False)
        self.kv_norm = nn.LayerNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=config.bias if hasattr(config, 'bias') else False)
        
        # Selective attention head
        self.selection_head_idx = getattr(config, 'selection_head_idx', 0)
        
        # Output projection
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=config.bias if hasattr(config, 'bias') else False)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        
        # Attention scaling factor
        self.softmax_scale = self.qk_head_dim ** -0.5
        
        # RoPE
        self.max_seq_len = getattr(config, 'max_seq_len', 4096)
        self.rope = RoPE(self.qk_rope_head_dim, self.max_seq_len)
        
        # Cache for storing pruning indices (for residual connections)
        self.register_buffer('pruning_indices', None, persistent=False)
        self.register_buffer('pruning_mask', None, persistent=False)
    
    def _compute_selection_scores(self, q_sdpa, k_sdpa):
        """Compute selection scores for pruning."""
        # Extract selection head
        q_select = q_sdpa[:, self.selection_head_idx:self.selection_head_idx+1]
        k_select = k_sdpa[:, self.selection_head_idx:self.selection_head_idx+1]
        
        # Compute attention scores
        S = torch.matmul(q_select, k_select.transpose(-2, -1)) * self.softmax_scale
        S = S.squeeze(1)  # [B, S, T]
        
        # Apply constraints
        S = F.relu(S)
        S[:, :, 0] = 0  # Preserve BOS
        
        # Zero diagonal
        diagonal_stride = S.stride(1) + S.stride(2)
        S.view(-1)[::diagonal_stride] = 0
        
        # Accumulate selection scores
        selection_scores = torch.cumsum(S, dim=-2)  # [B, S, T]
        
        # Average across target dimension to get per-query-token importance
        token_importance = selection_scores.mean(dim=-1)  # [B, S]
        
        return token_importance, selection_scores
    
    def _prune_tokens(self, tensors_dict, token_importance, bsz, seqlen):
        """
        Prune tokens based on importance scores.
        
        Args:
            tensors_dict: Dictionary of tensors to prune
            token_importance: Importance scores [B, S]
            bsz: Batch size
            seqlen: Sequence length
            
        Returns:
            pruned_dict: Dictionary of pruned tensors
            indices: Selected token indices
        """
        # Determine number of tokens to keep
        num_keep = max(int(seqlen * (1 - self.pruning_ratio)), self.min_kept_tokens)
        num_keep = min(num_keep, seqlen)
        
        # Get top-k indices
        _, indices = torch.topk(token_importance, num_keep, dim=1, sorted=True)
        
        # Sort indices to maintain relative order
        indices, _ = torch.sort(indices, dim=1)
        
        # Create gathering indices
        batch_indices = torch.arange(bsz, device=indices.device).unsqueeze(1).expand(-1, num_keep)
        
        # Prune tensors
        pruned_dict = {}
        for name, tensor in tensors_dict.items():
            if name == 'selection_scores':
                # Selection scores are [B, S, T] - only prune S dimension
                pruned_dict[name] = tensor[batch_indices, indices]
            elif tensor.dim() == 3:  # [B, S, D]
                pruned_dict[name] = tensor[batch_indices, indices]
            elif tensor.dim() == 4:  # [B, H, S, D] 
                # Need to handle head dimension
                pruned_dict[name] = tensor.transpose(1, 2)[batch_indices, indices].transpose(1, 2)
        
        return pruned_dict, indices
    
    def _restore_pruned_output(self, pruned_output, indices, bsz, original_seqlen):
        """Restore pruned output to original sequence length with zeros for pruned positions."""
        batch_indices = torch.arange(bsz, device=indices.device).unsqueeze(1).expand(-1, indices.size(1))
        
        # Create output tensor filled with zeros
        output = torch.zeros(bsz, original_seqlen, pruned_output.size(-1), 
                           device=pruned_output.device, dtype=pruned_output.dtype)
        
        # Scatter pruned values back
        output[batch_indices, indices] = pruned_output
        
        # Store pruning info for residual connection
        self.pruning_indices = indices
        mask = torch.zeros(bsz, original_seqlen, device=indices.device, dtype=torch.bool)
        mask[batch_indices, indices] = True
        self.pruning_mask = mask
        
        return output
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        """Forward pass with dynamic token pruning."""
        bsz, seqlen, _ = x.size()
        original_seqlen = seqlen
        
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
        
        # Process keys and values
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        
        # Apply RoPE to k_pe
        k_pe_reshaped = k_pe.view(bsz, seqlen, 1, -1).transpose(1, 2)
        k_pe_encoded = self.rope(k_pe_reshaped.contiguous(), start_pos).squeeze(1)
        
        # Compute full keys for selection scoring
        q_full = torch.cat([q_nope, q_pe], dim=-1)
        kv_norm = self.kv_norm(kv)
        kv_proj = self.wkv_b(kv_norm)
        kv_proj = kv_proj.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_proj, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_full = torch.cat([k_nope, k_pe_encoded.unsqueeze(2).expand(-1, -1, self.n_heads, -1)], dim=-1)
        
        # Prepare for attention
        q_sdpa = q_full.transpose(1, 2)  # [B, H, S, D]
        k_sdpa = k_full.transpose(1, 2)  # [B, H, T, D]
        v_sdpa = v.transpose(1, 2)  # [B, H, T, D]
        
        # Compute selection scores
        token_importance, selection_scores = self._compute_selection_scores(q_sdpa, k_sdpa)
        
        # Prune tokens if sequence is long enough
        if seqlen > self.min_kept_tokens * 2 and self.training:
            # Prepare tensors for pruning
            tensors_to_prune = {
                'q_sdpa': q_sdpa,
                'k_sdpa': k_sdpa,
                'v_sdpa': v_sdpa,
                'selection_scores': selection_scores
            }
            
            # Prune tokens
            pruned_tensors, indices = self._prune_tokens(tensors_to_prune, token_importance, bsz, seqlen)
            
            # Extract pruned tensors
            q_sdpa_pruned = pruned_tensors['q_sdpa']
            k_sdpa_pruned = pruned_tensors['k_sdpa']
            v_sdpa_pruned = pruned_tensors['v_sdpa']
            selection_scores_pruned = pruned_tensors['selection_scores']
            
            # Update sequence length
            pruned_seqlen = indices.size(1)
            
            # For pruned sequences, we need to compute selective attention properly
            # The selection scores were computed on the original sequence, so we need to prune them correctly
            # selection_scores_pruned is [B, S', T] but we need [B, S', T'] for attention
            
            # Extract the pruned selection scores for both query and key dimensions
            batch_indices_expanded = batch_indices.unsqueeze(-1).expand(-1, -1, pruned_seqlen)
            indices_expanded = indices.unsqueeze(1).expand(-1, pruned_seqlen, -1)
            
            # Gather selection scores for pruned sequences on both dimensions
            selection_scores_pruned_full = selection_scores_pruned.gather(2, indices_expanded)  # [B, S', T']
            selective_mask = selection_scores_pruned_full.unsqueeze(1)  # [B, 1, S', T']
            
            # Create causal mask for pruned sequence
            if mask is None and pruned_seqlen > 1:
                causal_mask = torch.triu(
                    torch.full((pruned_seqlen, pruned_seqlen), float('-inf'), 
                             device=x.device, dtype=q_sdpa_pruned.dtype),
                    diagonal=1
                )
                attn_mask = causal_mask.unsqueeze(0).unsqueeze(0) - selective_mask
            else:
                attn_mask = -selective_mask
            
            # Compute attention on pruned tokens
            attn_output = F.scaled_dot_product_attention(
                q_sdpa_pruned, k_sdpa_pruned, v_sdpa_pruned,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
                scale=self.softmax_scale
            )
            
            # Reshape
            x_pruned = attn_output.transpose(1, 2).contiguous()
            x_pruned = x_pruned.reshape(bsz, pruned_seqlen, -1)
            
            # Project to output dimension
            x_pruned = self.wo(x_pruned)
            
            # Restore to original sequence length
            x = self._restore_pruned_output(x_pruned, indices, bsz, original_seqlen)
            
        else:
            # Standard selective attention without pruning
            selective_mask = selection_scores.unsqueeze(1)
            
            if mask is None and seqlen > 1:
                causal_mask = torch.triu(
                    torch.full((seqlen, seqlen), float('-inf'), 
                             device=x.device, dtype=q_sdpa.dtype),
                    diagonal=1
                )
                attn_mask = causal_mask.unsqueeze(0).unsqueeze(0) - selective_mask
            else:
                attn_mask = -selective_mask if mask is None else mask.unsqueeze(0).unsqueeze(0) - selective_mask
            
            # Compute attention
            attn_output = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
                scale=self.softmax_scale
            )
            
            # Reshape and project
            x = attn_output.transpose(1, 2).contiguous()
            x = x.reshape(bsz, seqlen, -1)
            x = self.wo(x)
            
            # Clear pruning info
            self.pruning_indices = None
            self.pruning_mask = None
        
        # Handle FP8 conversion if needed
        if x.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            x = x.to(torch.bfloat16)
        
        x = self.resid_dropout(x)
        
        return x