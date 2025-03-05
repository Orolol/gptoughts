"""Expert modules for the LLaDA model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..blocks.moe import SharedExpertMLP, ExpertGroup

# LLaDAExpert peut rester spécifique car il implémente un mécanisme d'adaptation particulier
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
            
            # Adaptation pathway
            adapt_in = self.adapt_norm(self.pre_adapt(x))
            adapt_out = self.adapt_norm(self.post_adapt(hidden))
            
            # Compute adaptation weights
            adapt_weights = torch.bmm(adapt_in, adapt_out.transpose(1, 2))
            adapt_weights = torch.clamp(adapt_weights, -5.0, 5.0)
            adapt_weights = F.silu(adapt_weights)
            
            # Apply adaptation
            adapt = torch.bmm(adapt_weights, adapt_in)
            adapt = self.adapt_proj(adapt)
            
            # Scale adaptation contribution
            adapt = 0.1 * adapt
            
            # Combine paths
            hidden = hidden + adapt
            
            out = self.dropout(self.shared_down(hidden))
            
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


class LLaDAExpertGroup(ExpertGroup):
    """
    Expert group for LLaDA that extends the base ExpertGroup.
    """
    def __init__(self, config):
        # We'll use a custom initialization instead of calling super().__init__()
        # because we need to use LLaDAExpert instead of SharedExpertMLP
        nn.Module.__init__(self)
        self.num_experts = config.num_experts
        self.config = config
        
        # Define dimensions
        self.hidden_dim = 2 * config.n_embd
        self.adapt_dim = self.hidden_dim // 16
        
        # Use LLaDAExpert as the shared MLP backbone
        self.shared_mlp = LLaDAExpert(config)
        
        # Expert-specific adaptation - same as base ExpertGroup
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
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            # Make sure input is contiguous
            if not x.is_contiguous():
                x = x.contiguous()
                
            # Compute shared MLP output
            shared_output = self.shared_mlp(x)
            
            if self.parallel_adapters:
                # Process all experts in parallel
                if expert_weights.dim() == 2:
                    expert_weights = expert_weights.view(batch_size, seq_len, -1)
                
                # Create output tensor with optimal memory layout
                combined_output = torch.zeros_like(shared_output)
                
                # Process each expert
                for i in range(self.num_experts):
                    # Get mask for this expert
                    expert_mask = expert_weights[:, :, i] > 0
                    if not expert_mask.any():
                        continue
                    
                    # Get tokens for this expert
                    expert_x = x[expert_mask]
                    
                    # Skip if no tokens
                    if expert_x.numel() == 0:
                        continue
                    
                    # Forward through expert-specific adapter
                    expert_hidden = self.shared_mlp.pre_adapt(expert_x)
                    adapted = self.expert_adapters[i](expert_hidden)
                    
                    # Project through shared projections
                    expert_proj_output = self.expert_proj(adapted)
                    expert_output = self.output_proj(expert_proj_output)
                    
                    # Add to output tensor
                    combined_output[expert_mask] = expert_output
            
            # Add expert outputs to shared output
            return shared_output + 0.1 * combined_output 