"""Mixture of Experts components for transformer models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    """
    Router module that determines which expert should process each token.
    Uses top-k routing with load balancing.
    """
    def __init__(self, input_dim: int, num_experts: int, k: int = 2, capacity_factor: float = 1.25):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor
        
        # Simpler router architecture with better initialization
        self.router = nn.Sequential(
            nn.Linear(input_dim, num_experts, bias=False),
            nn.LayerNorm(num_experts)  # Normalize router logits
        )
        
        # Better initialization for routing
        nn.init.normal_(self.router[0].weight, mean=0.0, std=0.01)
        
        # Increased temperature for softer routing decisions
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        # Adjusted loss coefficients
        self.router_z_loss_coef = 0.01  # Increased from 0.001
        self.load_balance_coef = 0.01   # Increased from 0.001

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        combined_batch_size = batch_size * seq_len
        
        # Compute router logits
        router_logits = self.router(x.view(-1, self.input_dim))
        
        # Scale logits by learned temperature
        router_logits = router_logits / (self.temperature.abs() + 1e-6)
        
        # Compute routing probabilities with stable softmax
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Get top-k routing weights and indices
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.k, dim=-1)
        
        # Normalize top-k weights
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Create dispatch mask.
        dispatch_mask = torch.zeros_like(routing_weights)
        dispatch_mask.scatter_(
            dim=-1,
            index=top_k_indices,
            src=top_k_weights
        )
        
        # Compute load balancing loss
        # Ideal load would be uniform distribution across experts
        ideal_load = torch.ones_like(routing_weights.mean(0)) / self.num_experts
        actual_load = routing_weights.mean(0)
        load_balance_loss = F.kl_div(
            actual_load.log(),
            ideal_load,
            reduction='batchmean',
            log_target=False
        )
        
        # Router z-loss to encourage exploration
        router_z_loss = torch.square(router_logits).mean()
        
        # Combine losses with adjusted coefficients
        router_loss = (
            self.router_z_loss_coef * router_z_loss +
            self.load_balance_coef * load_balance_loss
        )
        
        # Reshape dispatch mask for batch processing
        dispatch_mask = dispatch_mask.view(batch_size, seq_len, -1)
        
        return routing_weights.detach(), dispatch_mask, router_loss

class SharedExpertMLP(nn.Module):
    """
    Shared MLP backbone for all experts with efficient computation.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Define dimensions with increased capacity
        self.hidden_dim = 4 * config.n_embd
        self.adapt_dim = self.hidden_dim // 16
        
        # Up projection with parallel computation
        self.up_proj = nn.Linear(config.n_embd, self.hidden_dim, bias=False)
        
        # Adapter projection for efficient computation
        self.adapt_proj = nn.Linear(self.hidden_dim, self.adapt_dim, bias=False)
        
        # Activation function
        self.activation = nn.SiLU()
        
        # Initialization with scaled values
        self._init_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Full forward pass through shared MLP
        projected = self.up_proj(x)
        activated = self.activation(projected)
        
        # No need for adapter projection in full forward
        return activated
    
    def pre_adapt(self, x: torch.Tensor) -> torch.Tensor:
        # Forward only up to adapter projection for expert-specific processing
        projected = self.up_proj(x)
        activated = self.activation(projected)
        adapter_input = self.adapt_proj(activated)
        return adapter_input
    
    def _init_weights(self):
        # Smaller initialization for more stability
        up_scale = 0.1 / (self.config.n_embd ** 0.5)
        adapt_scale = 0.1 / (self.hidden_dim ** 0.5)
        
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=up_scale)
        nn.init.normal_(self.adapt_proj.weight, mean=0.0, std=adapt_scale)

class ExpertGroup(nn.Module):
    """
    A group of experts that share some weights but maintain unique characteristics
    through adaptation layers.
    """
    def __init__(self, config, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.config = config
        
        # Define dimensions
        self.hidden_dim = 2 * config.n_embd
        self.adapt_dim = self.hidden_dim // 16
        
        # Shared MLP backbone with larger capacity
        self.shared_mlp = SharedExpertMLP(config)
        
        # Expert-specific adaptation with parallel computation
        self.expert_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.adapt_dim, self.adapt_dim, bias=False),
                nn.LayerNorm(self.adapt_dim)
            ) for _ in range(num_experts)
        ])
        
        # Projections for dimension matching
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
        shared_output = self.shared_mlp(x)
        
        if self.parallel_adapters:
            # Process all experts in parallel
            expert_outputs = []
            masks = []
            
            # Ensure expert_weights has correct shape [batch_size, seq_len, num_experts]
            if expert_weights.dim() == 2:
                expert_weights = expert_weights.view(batch_size, seq_len, -1)
            
            # Create masks for each expert
            for i in range(self.num_experts):
                mask = expert_weights[:, :, i] > 0
                if mask.any():
                    masks.append(mask)
                    expert_x = x[mask]
                    # Process through adapter and projections
                    expert_hidden = self.shared_mlp.pre_adapt(expert_x)
                    expert_hidden = self.expert_adapters[i](expert_hidden)
                    expert_hidden = self.expert_proj(expert_hidden)
                    expert_hidden = self.output_proj(expert_hidden)
                    expert_outputs.append(expert_hidden)
            
            # Combine expert outputs efficiently
            if expert_outputs:
                combined_output = torch.zeros_like(shared_output)
                for mask, expert_output in zip(masks, expert_outputs):
                    combined_output[mask] = expert_output
                
                return shared_output + 0.1 * combined_output
        
        return shared_output

class MoELayer(nn.Module):
    """
    Mixture of Experts layer that combines multiple expert MLPs with a router.
    """
    def __init__(self, config, num_experts: int = 8, k: int = 2):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.k = k
        
        # Initialize router with adjusted parameters
        self.router = Router(
            input_dim=config.n_embd,
            num_experts=num_experts,
            k=k,
            capacity_factor=1.5  # Increased capacity factor
        )
        
        # Expert group with shared parameters
        self.expert_group = ExpertGroup(config, num_experts)
        
        # Add layer norm before routing
        self.norm = nn.LayerNorm(config.n_embd)
        
        # Add residual scaling
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        
        # Normalize input
        normalized = self.norm(x)
        
        # Get routing weights and dispatch mask
        routing_weights, dispatch_mask, router_loss = self.router(normalized)
        
        # Process through expert group
        expert_output = self.expert_group(normalized, dispatch_mask)
        
        # Add scaled residual connection
        output = residual + self.residual_scale * expert_output
        
        return output, router_loss 