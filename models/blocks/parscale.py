"""
ParScale components for Multi-head Latent Attention (MLA).
Implementation based on the ParScale paper for parallel scaling of LLMs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class LatentPrefixMLA(nn.Module):
    """Learnable prefixes in the latent space for MLA."""
    
    def __init__(self, P: int, d_model: int, d_latent: int, prefix_length: int):
        super().__init__()
        self.P = P
        self.d_model = d_model
        self.d_latent = d_latent
        self.prefix_length = prefix_length
        
        # Initialize latent prefixes for Q, K, V
        self.latent_prefixes_q = nn.Parameter(self._init_prefixes(P, prefix_length, d_latent))
        self.latent_prefixes_k = nn.Parameter(self._init_prefixes(P, prefix_length, d_latent))
        self.latent_prefixes_v = nn.Parameter(self._init_prefixes(P, prefix_length, d_latent))
        
        # Optionally, prefixes in input space
        self.input_prefixes = nn.Parameter(
            torch.randn(P, prefix_length, d_model) * 0.02
        )
    
    def _init_prefixes(self, P: int, prefix_length: int, d_latent: int) -> torch.Tensor:
        """Initialize prefixes with orthogonalization for diversity."""
        std = math.sqrt(2.0 / (d_latent + prefix_length))
        prefixes = torch.randn(P, prefix_length, d_latent) * std
        
        # Orthogonalize using modified Gram-Schmidt
        for i in range(P):
            if i > 0:
                for j in range(i):
                    # Project out previous components
                    prefixes[i] -= torch.sum(prefixes[i] * prefixes[j]) * prefixes[j]
                # Normalize
                prefixes[i] = F.normalize(prefixes[i], dim=-1)
        
        return prefixes
    
    def get_prefixes(self, stream_idx: int) -> Dict[str, torch.Tensor]:
        """Get prefixes for a specific stream."""
        return {
            'q': self.latent_prefixes_q[stream_idx],
            'k': self.latent_prefixes_k[stream_idx],
            'v': self.latent_prefixes_v[stream_idx],
            'input': self.input_prefixes[stream_idx]
        }
    
    def get_all_prefixes(self) -> Dict[str, torch.Tensor]:
        """Get all prefixes."""
        return {
            'q': self.latent_prefixes_q,
            'k': self.latent_prefixes_k,
            'v': self.latent_prefixes_v,
            'input': self.input_prefixes
        }


class DynamicAggregator(nn.Module):
    """Dynamic aggregation of parallel streams with label smoothing."""
    
    def __init__(self, d_model: int, P: int, epsilon: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.P = P
        self.epsilon = epsilon
        
        # MLP for computing aggregation weights
        self.mlp = nn.Sequential(
            nn.Linear(d_model * P, d_model),
            nn.ReLU(),
            nn.Linear(d_model, P)
        )
    
    def forward(self, parallel_outputs: torch.Tensor) -> torch.Tensor:
        """
        Aggregate parallel outputs.
        
        Args:
            parallel_outputs: [batch_size, P, seq_len, d_model]
            
        Returns:
            aggregated: [batch_size, seq_len, d_model]
        """
        batch_size, P, seq_len, d_model = parallel_outputs.shape
        
        # Reshape for MLP input: [batch_size, seq_len, P * d_model]
        concat = parallel_outputs.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        # Compute weights: [batch_size, seq_len, P]
        weights = self.mlp(concat).softmax(dim=-1)
        
        # Apply label smoothing to prevent collapse
        weights = weights * (1 - self.epsilon) + self.epsilon / self.P
        
        # Weighted aggregation: [batch_size, seq_len, d_model]
        parallel_outputs = parallel_outputs.permute(0, 2, 1, 3)  # [B, L, P, D]
        aggregated = (parallel_outputs * weights.unsqueeze(-1)).sum(dim=2)
        
        return aggregated


class ParallelMLACache:
    """Parallel KV cache for MLA with multiple streams."""
    
    def __init__(self, P: int, max_length: int, d_latent: int, device: torch.device):
        self.P = P
        self.max_length = max_length
        self.d_latent = d_latent
        self.device = device
        
        # Initialize caches
        self.k_cache = torch.zeros(P, max_length, d_latent, device=device)
        self.v_cache = torch.zeros(P, max_length, d_latent, device=device)
        self.cache_pos = 0
    
    def update(self, k_latent: torch.Tensor, v_latent: torch.Tensor, stream_idx: int):
        """Update cache for a specific stream."""
        seq_len = k_latent.size(1)
        
        if self.cache_pos + seq_len > self.max_length:
            # Handle cache overflow
            self.cache_pos = 0
        
        self.k_cache[stream_idx, self.cache_pos:self.cache_pos + seq_len] = k_latent.squeeze(0)
        self.v_cache[stream_idx, self.cache_pos:self.cache_pos + seq_len] = v_latent.squeeze(0)
        
        return self.cache_pos, self.cache_pos + seq_len
    
    def get_cache(self, stream_idx: int, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached KV for a specific stream and position range."""
        return (
            self.k_cache[stream_idx, start:end],
            self.v_cache[stream_idx, start:end]
        )
    
    def clear(self):
        """Clear all caches."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_pos = 0


def diversity_loss(outputs: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Compute diversity loss to encourage different behaviors across streams.
    
    Args:
        outputs: [batch_size, P, seq_len, d_model]
        epsilon: Small constant for numerical stability
        
    Returns:
        diversity_penalty: Scalar tensor
    """
    batch_size, P, seq_len, d_model = outputs.shape
    
    # Normalize outputs
    normalized = F.normalize(outputs, dim=-1, eps=epsilon)
    
    # Reshape for similarity computation: [batch_size * seq_len, P, d_model]
    normalized = normalized.permute(0, 2, 1, 3).reshape(-1, P, d_model)
    
    # Compute cosine similarity matrix: [batch_size * seq_len, P, P]
    similarity_matrix = torch.bmm(normalized, normalized.transpose(1, 2))
    
    # Create mask for off-diagonal elements
    mask = 1 - torch.eye(P, device=outputs.device).unsqueeze(0)
    
    # Compute mean similarity (excluding diagonal)
    diversity_penalty = (similarity_matrix * mask).sum() / (mask.sum() * batch_size * seq_len)
    
    return diversity_penalty


class ComplexityEstimator(nn.Module):
    """Estimate input complexity for dynamic inference."""
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.complexity_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Estimate complexity of input.
        
        Args:
            input_ids: [batch_size, seq_len]
            
        Returns:
            complexity: [batch_size] values between 0 and 1
        """
        # Get embeddings
        embeddings = self.embedding(input_ids)
        
        # Pool over sequence dimension
        pooled = embeddings.mean(dim=1)
        
        # Compute complexity score
        complexity = self.complexity_head(pooled).squeeze(-1)
        
        return complexity