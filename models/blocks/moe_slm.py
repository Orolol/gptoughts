"""
Optimized Mixture of Experts (MoE) components for Small Language Models.

This module provides ultra-efficient MoE implementations with:
- Ultra-high weight sharing (90% shared, 10% specialized)
- Improved load balancing for many experts
- Optimized routing with temperature scaling
- Memory-efficient expert processing
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class SLMMoEConfig:
    """Configuration for SLM MoE components."""
    n_embd: int = 768
    n_inner: int = 2304  # 3 * n_embd for SLM
    num_experts: int = 32
    experts_per_token: int = 4
    shared_weight_ratio: float = 0.90
    router_temperature: float = 0.1
    router_z_loss_coef: float = 0.001
    load_balance_coef: float = 0.01
    dropout: float = 0.0
    bias: bool = False


class SLMRouter(nn.Module):
    """
    Optimized router for SLM with many experts and improved load balancing.
    
    Features:
    - Learnable temperature scaling
    - Advanced load balancing loss
    - Gumbel noise for exploration during training
    - Expert utilization tracking
    """
    
    def __init__(self, config: SLMMoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        
        # Router projection with layer normalization for stability
        self.router_norm = nn.LayerNorm(config.n_embd, eps=1e-6)
        self.router_proj = nn.Linear(config.n_embd, config.num_experts, bias=config.bias)
        
        # Learnable temperature with constraints
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(config.router_temperature)))
        
        # Loss coefficients
        self.router_z_loss_coef = config.router_z_loss_coef
        self.load_balance_coef = config.load_balance_coef
        
        # Expert utilization tracking (for monitoring)
        self.register_buffer('expert_counts', torch.zeros(config.num_experts))
        self.register_buffer('total_tokens', torch.zeros(1))
        
        # Dropout for router regularization
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize router weights with careful scaling
        nn.init.normal_(self.router_proj.weight, mean=0.0, std=0.01)
        if config.bias:
            nn.init.zeros_(self.router_proj.bias)
    
    @property
    def temperature(self) -> torch.Tensor:
        """Get temperature with constraints to avoid extreme values."""
        return torch.clamp(torch.exp(self.log_temperature), min=0.01, max=2.0)
    
    def forward(self, x: torch.Tensor, add_noise: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts with improved load balancing.
        
        Args:
            x: Input tensor [batch_size, seq_len, n_embd]
            add_noise: Whether to add Gumbel noise for exploration
            
        Returns:
            expert_indices: Selected expert indices [batch_size, seq_len, experts_per_token]
            expert_weights: Normalized weights for selected experts [batch_size, seq_len, experts_per_token]
            router_loss: Combined router loss (z-loss + load balancing)
        """
        batch_size, seq_len, n_embd = x.shape
        original_shape = x.shape
        
        # Normalize input for stability
        x_norm = self.router_norm(x)
        
        # Apply dropout for regularization
        if self.training:
            x_norm = self.dropout(x_norm)
        
        # Compute router logits
        router_logits = self.router_proj(x_norm)  # [batch_size, seq_len, num_experts]
        
        # Add Gumbel noise for exploration during training
        if self.training and add_noise:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(router_logits) + 1e-8) + 1e-8)
            router_logits = router_logits + gumbel_noise * 0.1
        
        # Apply temperature scaling
        router_logits_scaled = router_logits / self.temperature
        
        # Compute routing probabilities with numerical stability
        router_probs = F.softmax(router_logits_scaled, dim=-1)
        
        # Select top-k experts with improved selection
        expert_weights, expert_indices = torch.topk(
            router_probs, self.experts_per_token, dim=-1, largest=True, sorted=True
        )  # [batch_size, seq_len, experts_per_token]
        
        # Renormalize weights for selected experts
        expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Update expert utilization statistics
        if self.training:
            self._update_expert_stats(expert_indices, batch_size * seq_len)
        
        # Compute auxiliary losses
        router_loss = self._compute_router_losses(router_logits, router_probs)
        
        return expert_indices, expert_weights, router_loss
    
    def _update_expert_stats(self, expert_indices: torch.Tensor, num_tokens: int):
        """Update expert utilization statistics for monitoring."""
        with torch.no_grad():
            # Count how many times each expert is selected
            flat_indices = expert_indices.view(-1)
            for expert_id in range(self.num_experts):
                count = (flat_indices == expert_id).sum().float()
                self.expert_counts[expert_id] += count
            
            self.total_tokens += num_tokens * self.experts_per_token
    
    def get_expert_utilization(self) -> torch.Tensor:
        """Get normalized expert utilization statistics."""
        if self.total_tokens > 0:
            return self.expert_counts / self.total_tokens
        else:
            return torch.zeros_like(self.expert_counts)
    
    def reset_expert_stats(self):
        """Reset expert utilization statistics."""
        self.expert_counts.zero_()
        self.total_tokens.zero_()
    
    def _compute_router_losses(self, router_logits: torch.Tensor, 
                              router_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute comprehensive auxiliary losses for router training.
        
        Args:
            router_logits: Raw router logits [batch_size, seq_len, num_experts]
            router_probs: Softmax probabilities [batch_size, seq_len, num_experts]
        """
        # Router z-loss: encourages router logits to stay small
        z_loss = torch.square(router_logits).mean()
        
        # Load balancing loss: encourages uniform distribution across experts
        # Average probability mass per expert across all tokens
        expert_probs = router_probs.mean(dim=(0, 1))  # [num_experts]
        
        # Ideal uniform distribution
        ideal_prob = 1.0 / self.num_experts
        
        # Load balancing loss using squared error (more stable than KL divergence)
        load_balance_loss = torch.square(expert_probs - ideal_prob).sum()
        
        # Additional entropy regularization to encourage exploration
        entropy_loss = -torch.sum(expert_probs * torch.log(expert_probs + 1e-8))
        entropy_target = -math.log(1.0 / self.num_experts)  # Maximum entropy
        entropy_regularization = torch.square(entropy_loss - entropy_target)
        
        # Combine losses
        total_loss = (
            self.router_z_loss_coef * z_loss +
            self.load_balance_coef * (load_balance_loss + 0.1 * entropy_regularization)
        )
        
        return total_loss


class SLMSharedExpertMLP(nn.Module):
    """
    Ultra-efficient MLP with 90% shared weights for SLM.
    
    Architecture:
    - 90% of parameters are shared across all experts
    - 10% of parameters are expert-specific
    - Learnable mixing coefficients for optimal combination
    - Memory-efficient processing with selective computation
    """
    
    def __init__(self, config: SLMMoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.shared_ratio = config.shared_weight_ratio
        
        # Calculate dimensions
        self.hidden_dim = config.n_inner
        self.shared_dim = int(self.hidden_dim * self.shared_ratio)
        self.expert_dim = self.hidden_dim - self.shared_dim
        
        # Shared components (90% of parameters) - used by all experts
        self.shared_up_proj = nn.Linear(config.n_embd, self.shared_dim, bias=config.bias)
        self.shared_down_proj = nn.Linear(self.shared_dim, config.n_embd, bias=config.bias)
        
        # Expert-specific components (10% of parameters per expert)
        self.expert_up_proj = nn.ModuleList([
            nn.Linear(config.n_embd, self.expert_dim, bias=config.bias)
            for _ in range(config.num_experts)
        ])
        self.expert_down_proj = nn.ModuleList([
            nn.Linear(self.expert_dim, config.n_embd, bias=config.bias)
            for _ in range(config.num_experts)
        ])
        
        # Learnable mixing coefficients per expert
        # Each expert learns how to combine shared and specialized outputs
        self.mixing_coeffs = nn.Parameter(
            torch.zeros(config.num_experts, 3)  # [shared_weight, expert_weight, bias]
        )
        
        # Activation function
        self.activation = nn.SiLU()
        
        # Dropout for regularization
        if config.dropout > 0:
            self.dropout = nn.Dropout(config.dropout)
        else:
            self.dropout = nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor, 
                expert_weights: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with efficient expert routing.
        
        Args:
            x: Input tensor [batch_size, seq_len, n_embd]
            expert_indices: Selected expert indices [batch_size, seq_len, experts_per_token]
            expert_weights: Weights for selected experts [batch_size, seq_len, experts_per_token]
        """
        batch_size, seq_len, n_embd = x.shape
        experts_per_token = expert_indices.shape[-1]
        
        # Shared processing (applied to all tokens)
        shared_hidden = self.activation(self.shared_up_proj(x))
        shared_hidden = self.dropout(shared_hidden)
        shared_output = self.shared_down_proj(shared_hidden)
        
        # Expert processing (applied selectively)
        # Use vectorized operations for efficiency
        output = torch.zeros_like(x)
        
        # Process each expert that's selected
        unique_experts = torch.unique(expert_indices)
        
        for expert_id in unique_experts:
            if expert_id >= self.num_experts:
                continue
            
            # Find all tokens that use this expert
            expert_mask = (expert_indices == expert_id)  # [batch_size, seq_len, experts_per_token]
            
            if not expert_mask.any():
                continue
            
            # Get the tokens and weights for this expert
            # Create a mask for tokens that use this expert
            token_mask = expert_mask.any(dim=-1)  # [batch_size, seq_len]
            
            if not token_mask.any():
                continue
            
            # Extract tokens that use this expert
            expert_tokens = x[token_mask]  # [num_tokens, n_embd]
            
            if expert_tokens.numel() == 0:
                continue
            
            # Process through expert-specific layers
            expert_hidden = self.activation(self.expert_up_proj[expert_id](expert_tokens))
            expert_hidden = self.dropout(expert_hidden)
            expert_specific_output = self.expert_down_proj[expert_id](expert_hidden)
            
            # Get shared output for these tokens
            expert_shared_output = shared_output[token_mask]
            
            # Combine shared and expert outputs using learnable mixing
            mixing_weights = torch.softmax(self.mixing_coeffs[expert_id], dim=-1)
            shared_weight = mixing_weights[0]
            expert_weight = mixing_weights[1]
            bias_weight = mixing_weights[2]
            
            # Mix outputs
            mixed_output = (
                shared_weight * expert_shared_output +
                expert_weight * expert_specific_output +
                bias_weight * expert_tokens  # Residual connection
            )
            
            # Gather weights for this expert
            expert_weight_mask = expert_mask
            expert_token_weights = expert_weights * expert_weight_mask.float()
            expert_token_weights = expert_token_weights.sum(dim=-1)  # [batch_size, seq_len]
            expert_token_weights = expert_token_weights[token_mask]  # [num_tokens]
            
            # Apply routing weights
            weighted_output = mixed_output * expert_token_weights.unsqueeze(-1)
            
            # Add to final output
            output[token_mask] += weighted_output
        
        return output
    
    def _init_weights(self):
        """Initialize weights with careful scaling for stability."""
        # Shared components get normal initialization
        std = math.sqrt(2.0 / (self.config.n_embd + self.shared_dim))
        nn.init.normal_(self.shared_up_proj.weight, mean=0.0, std=std)
        
        std = math.sqrt(2.0 / (self.shared_dim + self.config.n_embd))
        nn.init.normal_(self.shared_down_proj.weight, mean=0.0, std=std)
        
        # Expert components get smaller initialization
        expert_std = std * 0.1  # Much smaller for expert-specific parts
        
        for expert_id in range(self.num_experts):
            # Up projection
            nn.init.normal_(
                self.expert_up_proj[expert_id].weight, 
                mean=0.0, 
                std=math.sqrt(2.0 / (self.config.n_embd + self.expert_dim)) * 0.1
            )
            
            # Down projection
            nn.init.normal_(
                self.expert_down_proj[expert_id].weight, 
                mean=0.0, 
                std=math.sqrt(2.0 / (self.expert_dim + self.config.n_embd)) * 0.1
            )
        
        # Initialize biases to zero
        if self.config.bias:
            nn.init.zeros_(self.shared_up_proj.bias)
            nn.init.zeros_(self.shared_down_proj.bias)
            for expert_id in range(self.num_experts):
                nn.init.zeros_(self.expert_up_proj[expert_id].bias)
                nn.init.zeros_(self.expert_down_proj[expert_id].bias)
        
        # Initialize mixing coefficients to favor shared processing initially
        with torch.no_grad():
            self.mixing_coeffs[:, 0] = 2.0   # High weight for shared
            self.mixing_coeffs[:, 1] = -1.0  # Low weight for expert-specific
            self.mixing_coeffs[:, 2] = -2.0  # Low weight for residual


class SLMMoELayer(nn.Module):
    """
    Complete SLM MoE layer combining router and shared expert MLP.
    """
    
    def __init__(self, config: SLMMoEConfig):
        super().__init__()
        self.config = config
        
        # Router for expert selection
        self.router = SLMRouter(config)
        
        # Shared expert MLP
        self.mlp = SLMSharedExpertMLP(config)
        
        # Layer normalization
        self.norm = nn.LayerNorm(config.n_embd)
        
        # Residual scaling for stability
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, n_embd]
            
        Returns:
            output: Processed tensor [batch_size, seq_len, n_embd]
            router_loss: Auxiliary loss for router training
        """
        residual = x
        
        # Normalize input
        x_norm = self.norm(x)
        
        # Route tokens to experts
        expert_indices, expert_weights, router_loss = self.router(x_norm)
        
        # Process through MoE MLP
        mlp_output = self.mlp(x_norm, expert_indices, expert_weights)
        
        # Apply residual connection with learned scaling
        output = residual + self.residual_scale * mlp_output
        
        return output, router_loss
    
    def get_expert_utilization(self) -> torch.Tensor:
        """Get expert utilization statistics."""
        return self.router.get_expert_utilization()
    
    def reset_expert_stats(self):
        """Reset expert utilization statistics."""
        self.router.reset_expert_stats()


def create_slm_moe_config(n_embd: int = 768, num_experts: int = 32, 
                         experts_per_token: int = 4, **kwargs) -> SLMMoEConfig:
    """Create SLM MoE configuration with sensible defaults."""
    return SLMMoEConfig(
        n_embd=n_embd,
        n_inner=3 * n_embd,  # SLM uses 3x instead of 4x
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        **kwargs
    )