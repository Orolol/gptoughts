"""Native Sparse Attention (NSA) mechanisms for transformer models."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass



@dataclass
class NSAConfig:
    """Configuration for Native Sparse Attention."""
    # Model dimensions
    hidden_dim: int = 2560
    n_heads: int = 64
    n_groups: int = 4  # For GQA
    head_dim: int = 192
    value_dim: int = 128
    qk_rope_head_dim: int = 64
    
    # NSA parameters
    compress_block_size: int = 32      # l
    compress_stride: int = 16          # d
    selection_block_size: int = 64     # l'
    num_selected_blocks: int = 16      # n
    sliding_window_size: int = 512     # w
    
    # Optimizations
    use_fp8: bool = False
    use_fp4_inference: bool = False
    tensor_core_block_size: int = 128
    enable_microscaling: bool = False
    
    # Training
    dropout: float = 0.0
    bias: bool = False


class CompressionMLP(nn.Module):
    """MLP module for compressing blocks of tokens."""
    
    def __init__(self, config: NSAConfig):
        super().__init__()
        self.config = config
        
        # Position encoding for intra-block
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.compress_block_size, config.hidden_dim) * 0.02
        )
        
        # Compression MLP
        self.compress = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim, bias=config.bias),
            nn.LayerNorm(config.hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress sequences of tokens into representative tokens (Optimized, vectorized version).
        
        Args:
            x: Input tensor of shape [B, T, D]
            
        Returns:
            Compressed tokens of shape [B, num_blocks, D]
        """
        B, T, D = x.shape
        l, d = self.config.compress_block_size, self.config.compress_stride
        
        # Handle edge case where sequence is shorter than block size
        if T < l:
            x_with_pos = x + self.pos_embed[:, :T, :]
            compressed = self.compress(x_with_pos).mean(dim=1, keepdim=True)
            return compressed

        # Vectorized block creation using unfold
        # Create overlapping blocks of size l with stride d
        # x_unfolded: [B, D, num_blocks, l]
        x_unfolded = x.transpose(1, 2).unfold(2, l, d)
        
        # Permute to get [B, num_blocks, l, D]
        blocks = x_unfolded.permute(0, 2, 3, 1).contiguous()
        num_blocks = blocks.shape[1]

        # Add position embedding.
        # self.pos_embed shape is [1, l, D]. It will be broadcasted to [B, num_blocks, l, D].
        blocks = blocks + self.pos_embed

        # Reshape for MLP: [B * num_blocks, l, D]
        blocks_reshaped = blocks.reshape(-1, l, D)
        
        # Compress all blocks at once
        compressed_reshaped = self.compress(blocks_reshaped)
        
        # Take the mean and reshape back: [B * num_blocks, D] -> [B, num_blocks, D]
        compressed = compressed_reshaped.mean(dim=1).view(B, num_blocks, D)
        
        return compressed


class BlockwiseSelection(nn.Module):
    """Module for selecting important blocks of tokens."""
    
    def __init__(self, config: NSAConfig):
        super().__init__()
        self.config = config
        
    def compute_importance_scores(
        self, 
        q: torch.Tensor, 
        k_compressed: torch.Tensor,
        attn_scores_compressed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute importance scores for block selection.
        
        Args:
            q: Query tensor [B, H, T, head_dim]
            k_compressed: Compressed key tensor [B, H, num_blocks, head_dim]
            attn_scores_compressed: Attention scores from compression [B, H, T, num_blocks]
            
        Returns:
            Importance scores [B, n_groups, num_blocks]
        """
        B, H, T_q, _ = q.shape
        l_prime = self.config.selection_block_size
        d = self.config.compress_stride
        
        # Aggregate scores based on spatial relationship
        # Use conv1d for efficient implementation
        kernel_size = l_prime // d
        if kernel_size > 0:
            # Reshape for conv1d: [B*H, T, num_blocks] -> [B*H, num_blocks, T]
            scores = attn_scores_compressed.view(B * H, T_q, -1).transpose(1, 2)
            
            # Apply 1D convolution to aggregate scores
            num_blocks = scores.shape[1]
            kernel = torch.ones(num_blocks, 1, kernel_size, device=q.device, dtype=q.dtype) / kernel_size
            importance = F.conv1d(scores, kernel, stride=1, groups=num_blocks)
            importance = importance.transpose(1, 2)  # [B*H, T_conv, num_blocks]
            
            # Take mean over query positions
            importance = importance.mean(dim=1)  # [B*H, num_blocks]
            importance = importance.view(B, H, -1)
        else:
            # Fallback: simple mean aggregation
            importance = attn_scores_compressed.mean(dim=2)  # [B, H, num_blocks]
        
        # Aggregate across heads for group query attention
        heads_per_group = H // self.config.n_groups
        importance = importance.view(B, self.config.n_groups, heads_per_group, -1)
        importance = importance.sum(dim=2)  # [B, n_groups, num_blocks]
        
        return importance
    
    def select_top_blocks(self, importance: torch.Tensor, n: int) -> torch.Tensor:
        """
        Select top-n blocks based on importance scores.
        
        Args:
            importance: Importance scores [B, n_groups, num_blocks]
            n: Number of blocks to select
            
        Returns:
            Indices of selected blocks [B, n_groups, n]
        """
        num_available_blocks = importance.shape[-1]
        k = min(n, num_available_blocks)  # Don't select more blocks than available
        _, indices = torch.topk(importance, k, dim=-1, sorted=True)
        
        # If we have fewer blocks than requested, pad with zeros
        if k < n:
            B, n_groups = importance.shape[:2]
            padding = torch.zeros(B, n_groups, n - k, dtype=indices.dtype, device=indices.device)
            indices = torch.cat([indices, padding], dim=-1)
        
        return indices


class NSAAttention(nn.Module):
    """
    Native Sparse Attention module with three parallel branches:
    1. Compression branch for global context
    2. Selection branch for important fine-grained tokens
    3. Sliding window branch for local context
    """
    
    def __init__(self, config: NSAConfig):
        super().__init__()
        self.config = config
        
        # Dimensions
        self.hidden_dim = config.hidden_dim
        self.n_heads = config.n_heads
        self.n_groups = config.n_groups
        self.head_dim = config.head_dim
        self.value_dim = config.value_dim
        
        # Compression module
        self.compression_mlp = CompressionMLP(config)
        
        # Selection module
        self.block_selection = BlockwiseSelection(config)
        
        # QKV projections for each branch
        self.qkv_compressed = nn.Linear(
            self.hidden_dim, 
            self.n_heads * (2 * self.head_dim + self.value_dim),
            bias=config.bias
        )
        self.qkv_selected = nn.Linear(
            self.hidden_dim,
            self.n_heads * (2 * self.head_dim + self.value_dim),
            bias=config.bias
        )
        self.qkv_window = nn.Linear(
            self.hidden_dim,
            self.n_heads * (2 * self.head_dim + self.value_dim),
            bias=config.bias
        )
        
        # Gating mechanism
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4, bias=config.bias),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 4, 3, bias=config.bias),
            nn.Sigmoid()
        )
        
        # Output projection
        self.out_proj = nn.Linear(
            self.n_heads * self.value_dim,
            self.hidden_dim,
            bias=config.bias
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Scaling factor
        self.scale = math.sqrt(self.head_dim)
        
    def compressed_attention(
        self, 
        x: torch.Tensor,
        rope: nn.Module,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform attention on compressed tokens.
        
        Args:
            x: Input tensor [B, T, D]
            mask: Optional attention mask
            
        Returns:
            Attention output [B, T, n_heads * value_dim]
            Attention scores for importance computation [B, n_heads, T, num_blocks]
        """
        B, T, D = x.shape
        
        # Compress tokens
        x_compressed = self.compression_mlp(x)  # [B, num_blocks, D]
        num_blocks = x_compressed.shape[1]
        
        # QKV for compressed tokens
        qkv = self.qkv_compressed(x_compressed)
        qkv = qkv.view(B, num_blocks, self.n_heads, 2 * self.head_dim + self.value_dim)
        qkv = qkv.transpose(1, 2)  # [B, n_heads, num_blocks, 2*head_dim + value_dim]
        
        k_compressed, v_compressed = torch.split(
            qkv[..., self.head_dim:], 
            [self.head_dim, self.value_dim], 
            dim=-1
        )
        
        # Query from original sequence
        q = self.qkv_compressed(x)
        q = q.view(B, T, self.n_heads, 2 * self.head_dim + self.value_dim)
        q = q.transpose(1, 2)[..., :self.head_dim]  # [B, n_heads, T, head_dim]
        
        # Apply RoPE to query
        q_rope = q[..., :self.config.qk_rope_head_dim]
        q_rope = rope(q_rope)
        q = torch.cat((q_rope, q[..., self.config.qk_rope_head_dim:]), dim=-1)
        
        # Attention scores
        scores = torch.matmul(q, k_compressed.transpose(-2, -1)) / self.scale
        
        # --- Causal Masking ---
        # A query at position `t` can only attend to compressed blocks that were generated
        # entirely from tokens in the past. The end of a block must be <= t.
        q_indices = torch.arange(T, device=x.device).view(1, 1, T, 1)
        block_end_indices = (torch.arange(num_blocks, device=x.device) * self.config.compress_stride + self.config.compress_block_size).view(1, 1, 1, num_blocks)
        causal_mask = q_indices >= block_end_indices
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        
        # Apply mask if provided
        if mask is not None:
            # Adjust mask for compressed sequence
            compressed_mask = self._create_compressed_mask(mask, num_blocks)
            if compressed_mask is not None:
                scores = scores.masked_fill(compressed_mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        output = torch.matmul(attn_weights, v_compressed)  # [B, n_heads, T, value_dim]
        output = output.transpose(1, 2).contiguous()
        output = output.view(B, T, self.n_heads * self.value_dim)
        
        return output, scores
    
    def selected_attention(
        self,
        x: torch.Tensor,
        rope: nn.Module,
        attn_scores_compressed: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform attention on selected blocks with causal masking, using a group-wise approach for clarity and stability.
        
        Args:
            x: Input tensor [B, T, D]
            rope: RoPE module
            attn_scores_compressed: Attention scores from compression branch
            mask: Optional attention mask
            
        Returns:
            Attention output [B, T, n_heads * value_dim]
        """
        B, T, D = x.shape
        H, G = self.n_heads, self.n_groups
        heads_per_group = H // G
        l_prime, n_selected = self.config.selection_block_size, self.config.num_selected_blocks

        # QKV projection
        qkv = self.qkv_selected(x)
        qkv = qkv.view(B, T, H, 2 * self.head_dim + self.value_dim)
        q, k, v = torch.split(qkv, [self.head_dim, self.head_dim, self.value_dim], dim=-1)
        
        # Reshape and apply RoPE
        q = q.transpose(1, 2) # [B, H, T, head_dim]
        k = k.transpose(1, 2) # [B, H, T, head_dim]
        v = v.transpose(1, 2) # [B, H, T, value_dim]

        q_rope = rope(q[..., :self.config.qk_rope_head_dim])
        q = torch.cat((q_rope, q[..., self.config.qk_rope_head_dim:]), dim=-1)
        k_rope = rope(k[..., :self.config.qk_rope_head_dim])
        k = torch.cat((k_rope, k[..., self.config.qk_rope_head_dim:]), dim=-1)

        # Compute importance and select top blocks for each group
        k_compressed_dummy = k.mean(dim=2, keepdim=True)
        importance = self.block_selection.compute_importance_scores(q, k_compressed_dummy, attn_scores_compressed)
        selected_block_indices = self.block_selection.select_top_blocks(importance, n_selected) # [B, G, n_selected]

        # Process group by group to ensure correctness
        output_groups = []
        for g in range(G):
            # Q for this group: [B, H/G, T, head_dim]
            q_group = q.view(B, G, heads_per_group, T, self.head_dim)[:, g, ...]
            
            # Get K and V for this group's selected blocks
            group_block_indices = selected_block_indices[:, g, :] # [B, n_selected]
            
            # Convert block indices to token indices
            start_indices = (group_block_indices * self.config.compress_stride) # [B, n_selected]
            block_offsets = torch.arange(l_prime, device=x.device) # [l_prime]
            token_indices = start_indices.unsqueeze(-1) + block_offsets # [B, n_selected, l_prime]
            token_indices = torch.clamp(token_indices.view(B, -1), 0, T - 1) # [B, n_selected * l_prime]
            
            # Gather K and V for the group
            k_group = k.view(B, G, heads_per_group, T, self.head_dim)[:, g, ...] # [B, H/G, T, head_dim]
            v_group = v.view(B, G, heads_per_group, T, self.value_dim)[:, g, ...] # [B, H/G, T, value_dim]
            
            # Expand indices for gathering: [B, 1, n_sel*l', 1] -> [B, H/G, n_sel*l', dim]
            token_indices_expanded_k = token_indices.view(B, 1, -1, 1).expand(-1, heads_per_group, -1, self.head_dim)
            k_gathered = torch.gather(k_group, 2, token_indices_expanded_k) # [B, H/G, n_sel*l', head_dim]

            token_indices_expanded_v = token_indices.view(B, 1, -1, 1).expand(-1, heads_per_group, -1, self.value_dim)
            v_gathered = torch.gather(v_group, 2, token_indices_expanded_v) # [B, H/G, n_sel*l', value_dim]

            # Attention scores: [B, H/G, T, n_sel*l']
            scores = torch.matmul(q_group, k_gathered.transpose(-2, -1)) / self.scale

            # --- Causal Masking ---
            # Query at `t` can only see keys at or before `t`.
            q_indices = torch.arange(T, device=x.device).view(1, 1, T, 1)
            k_indices_gathered = token_indices.view(B, 1, 1, -1)
            causal_mask = q_indices >= k_indices_gathered # Broadcasts to [B, 1, T, n_sel*l']
            scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)

            # Softmax and attention
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            output_group = torch.matmul(attn_weights, v_gathered) # [B, H/G, T, value_dim]
            output_groups.append(output_group)

        # Concatenate group outputs and project
        output = torch.cat(output_groups, dim=1) # [B, H, T, value_dim]
        output = output.transpose(1, 2).contiguous().view(B, T, H * self.value_dim)
        return output
    
    def window_attention(
        self,
        x: torch.Tensor,
        rope: nn.Module,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform sliding window attention.
        
        Args:
            x: Input tensor [B, T, D]
            mask: Optional attention mask
            
        Returns:
            Attention output [B, T, n_heads * value_dim]
        """
        B, T, D = x.shape
        window_size = self.config.sliding_window_size
        
        # QKV projection
        qkv = self.qkv_window(x)
        qkv = qkv.view(B, T, self.n_heads, 2 * self.head_dim + self.value_dim)
        q, k, v = torch.split(
            qkv.transpose(1, 2),
            [self.head_dim, self.head_dim, self.value_dim],
            dim=-1
        )

        # Apply RoPE to query and key
        q_rope = q[..., :self.config.qk_rope_head_dim]
        q_rope = rope(q_rope)
        q = torch.cat((q_rope, q[..., self.config.qk_rope_head_dim:]), dim=-1)

        k_rope = k[..., :self.config.qk_rope_head_dim]
        k_rope = rope(k_rope)
        k = torch.cat((k_rope, k[..., self.config.qk_rope_head_dim:]), dim=-1)
        
        # Create sliding window mask (vectorized)
        q_indices = torch.arange(T, device=x.device)[:, None]
        k_indices = torch.arange(T, device=x.device)[None, :]
        
        window_mask = (k_indices <= q_indices) & (k_indices > q_indices - window_size)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply window mask
        scores = scores.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)
        
        # Try to use Flash Attention if available and appropriate
        if hasattr(F, 'scaled_dot_product_attention') and mask is None and T <= 2048:
            # Use Flash Attention for efficiency
            output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=window_mask.unsqueeze(0).unsqueeze(0).to(q.dtype),
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=False  # We handle causality with window_mask
            )
            output = output.transpose(1, 2).contiguous()
            output = output.view(B, T, self.n_heads * self.value_dim)
            return output
        
        # Apply additional mask if provided
        if mask is not None:
            # Handle both 2D and 4D masks
            if mask.dim() == 2:
                # Simple 2D mask [B, T] - expand to match scores shape
                mask_expanded = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T]
                scores = scores.masked_fill(mask_expanded == 0, float('-inf'))
            else:
                # 4D mask [B, H, T, T]
                scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous()
        output = output.view(B, T, self.n_heads * self.value_dim)
        
        return output
    
    def _create_compressed_mask(
        self, 
        mask: Optional[torch.Tensor], 
        num_blocks: int
    ) -> Optional[torch.Tensor]:
        """
        Create mask for compressed sequence.
        
        Args:
            mask: Original mask - can be:
                  - 2D: [B, T] where T is sequence length
                  - 4D: [B, H, T_q, T_k] attention mask
            num_blocks: Number of compressed blocks
            
        Returns:
            Compressed mask with shape:
            - If input is 2D: [B, 1, 1, num_blocks]
            - If input is 4D: [B, H, T_q, num_blocks]
        """
        if mask is None:
            return None
        
        B = mask.shape[0]
        l = self.config.compress_block_size
        d = self.config.compress_stride
        
        if mask.dim() == 2:
            T = mask.shape[1]
            # Handle edge case where sequence is shorter than the compression block size
            if T < l:
                # The single compressed block is valid if any token in the sequence is valid.
                compressed_mask = mask.any(dim=1, keepdim=True).float() # [B, 1]
                return compressed_mask.view(B, 1, 1, 1) # [B, 1, 1, num_blocks=1]

            # 2D mask [B, T] -> Unfold to create blocks
            mask_unfolded = mask.unfold(1, l, d) # [B, num_blocks, l]
            
            # Block is valid if it contains any valid tokens
            compressed_mask = mask_unfolded.any(dim=2).float() # [B, num_blocks]
            
            # Reshape for attention: [B, 1, 1, num_blocks]
            return compressed_mask.view(B, 1, 1, -1)
            
        elif mask.dim() == 4:
            T_k = mask.shape[3]
            # Handle edge case where sequence is shorter than the compression block size
            if T_k < l:
                # The single compressed block is valid if any token is attendable.
                compressed_mask = mask.any(dim=-1, keepdim=True).float() # [B, H, T_q, 1]
                return compressed_mask

            # 4D mask [B, H, T_q, T_k] - compress along T_k dimension
            # Using max pooling is a good alternative to unfolding a 4D tensor.
            
            # Pad mask to be divisible by stride
            padding = (d - (T_k - l) % d) % d
            padded_mask = F.pad(mask.float(), (0, padding), 'constant', 0)

            # Use max pooling to check if any value in a block is 1 (attendable)
            compressed_mask = F.max_pool2d(
                padded_mask, 
                kernel_size=(1, l), 
                stride=(1, d)
            )
            # Ensure the output has the right number of blocks
            if compressed_mask.shape[3] != num_blocks:
                 # Fallback to loop if pooling gives incorrect shape
                 _, H, T_q, _ = mask.shape
                 compressed_mask_fallback = torch.zeros(B, H, T_q, num_blocks, device=mask.device, dtype=mask.dtype)
                 for i in range(num_blocks):
                     start = i * d
                     end = min(start + l, T_k)
                     compressed_mask_fallback[:, :, :, i] = mask[:, :, :, start:end].any(dim=-1).float()
                 return compressed_mask_fallback
            
            return compressed_mask
        
        else:
            raise ValueError(f"Unsupported mask dimension: {mask.dim()}. Expected 2D or 4D.")
    
    def forward(
        self,
        x: torch.Tensor,
        rope: nn.Module,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of Native Sparse Attention.
        
        Args:
            x: Input tensor [B, T, D]
            mask: Optional attention mask [B, T] or [B, T, T]
            
        Returns:
            Output tensor [B, T, D]
        """
        B, T, D = x.shape
        
        # Compute gates
        gates = self.gate_mlp(x.mean(dim=1, keepdim=True))  # [B, 1, 3]
        gates = gates.expand(B, T, 3)
        
        # Three parallel attention branches
        attn_compressed, scores_compressed = self.compressed_attention(x, rope, mask)
        
        attn_selected = self.selected_attention(x, rope, scores_compressed, mask)
        
        attn_window = self.window_attention(x, rope, mask)
        
        # Weighted fusion
        output = (
            gates[..., 0:1] * attn_compressed +
            gates[..., 1:2] * attn_selected +
            gates[..., 2:3] * attn_window
        )
        
        # Output projection
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output
