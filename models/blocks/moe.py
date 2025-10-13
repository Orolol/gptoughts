"""Mixture of Experts components for transformer models."""

import math

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
        
        # Reduced loss coefficients for memory efficiency
        self.router_z_loss_coef = 0.001  # Small coefficient to reduce memory overhead
        self.load_balance_coef = 0.001   # Small coefficient to reduce memory overhead

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        # Compute router logits
        router_logits = self.router(x.reshape(batch_size * seq_len, self.input_dim))
        
        # Scale logits by learned temperature
        router_logits = router_logits / (self.temperature.abs() + 1e-6)
        
        # Compute routing probabilities with stable softmax
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Get top-k routing weights and indices
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.k, dim=-1)
        
        # Normalize top-k weights
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
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
        
        top_k_weights = top_k_weights.view(batch_size, seq_len, self.k)
        top_k_indices = top_k_indices.view(batch_size, seq_len, self.k)

        return top_k_weights, top_k_indices, router_loss

class SharedExpertMLP(nn.Module):
    """
    Shared MLP backbone for all experts with efficient computation.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Define dimensions (reduced for memory efficiency)
        self.hidden_dim = 4 * config.n_embd
        self.adapt_dim = self.hidden_dim // 32  # Reduced from //16 for less memory
        
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
        
        # Note: We intentionally return the full hidden dimension result.
        # The MLP expands the dimension from n_embd to hidden_dim (4x)
        # The downstream components should be designed to handle this expanded dimension.
        # We don't need to reshape it back to the input dimension here, as other
        # components like ExpertGroup will handle appropriate projections.
        return activated
    
    def pre_adapt(self, x: torch.Tensor) -> torch.Tensor:
        # Forward only up to adapter projection for expert-specific processing
        projected = self.up_proj(x)
        activated = self.activation(projected)
        
        # Ensure that activated has proper shape before applying adapt_proj
        # The adapter expects input with dimension self.hidden_dim
        if activated.shape[-1] != self.hidden_dim:
            if activated.shape[-1] < self.hidden_dim:
                # Pad to match hidden dimension
                pad_size = self.hidden_dim - activated.shape[-1]
                activated = torch.nn.functional.pad(activated, (0, pad_size))
            else:
                # Truncate to match hidden dimension
                activated = activated[..., :self.hidden_dim]
                
        adapter_input = self.adapt_proj(activated)
        return adapter_input
    
    def _init_weights(self):
        # Smaller initialization for more stability
        up_scale = 0.1 / (self.config.n_embd ** 0.5)
        adapt_scale = 0.1 / (self.hidden_dim ** 0.5)
        
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=up_scale)
        nn.init.normal_(self.adapt_proj.weight, mean=0.0, std=adapt_scale)

class ExpertGroup(nn.Module):
    """Expert group with shared trunk and vectorised expert-specific adapters."""

    def __init__(self, config, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.config = config

        # Shared trunk (common to all experts)
        self.shared_mlp = SharedExpertMLP(config)
        self.hidden_dim = self.shared_mlp.hidden_dim  # 4 * config.n_embd
        self.adapt_dim = self.shared_mlp.adapt_dim

        # Expert-specific adapter parameters stored as tensors for easy batching
        self.adapter_weight = nn.Parameter(torch.empty(num_experts, self.adapt_dim, self.adapt_dim))
        self.adapter_ln_weight = nn.Parameter(torch.ones(num_experts, self.adapt_dim))
        self.adapter_ln_bias = nn.Parameter(torch.zeros(num_experts, self.adapt_dim))
        self.norm_eps = 1e-5

        # Projections back to model dimension (shared across experts)
        self.expert_proj = nn.Linear(self.adapt_dim, self.hidden_dim, bias=False)
        self.output_proj = nn.Linear(self.hidden_dim, config.n_embd, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        adapt_scale = 0.01 / (self.adapt_dim ** 0.5)
        nn.init.kaiming_uniform_(self.adapter_weight, a=math.sqrt(5))
        nn.init.ones_(self.adapter_ln_weight)
        nn.init.zeros_(self.adapter_ln_bias)
        nn.init.normal_(self.expert_proj.weight, mean=0.0, std=adapt_scale)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=adapt_scale)

    def compute_shared(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute shared trunk outputs and adapter inputs."""
        shared_output = self.shared_mlp(x)            # [B, S, hidden_dim]
        shared_proj = self.output_proj(shared_output) # [B, S, n_embd]
        pre_adapt = self.shared_mlp.pre_adapt(x)      # [B, S, adapt_dim]
        return shared_proj, pre_adapt

    def mix_experts(
        self,
        pre_adapt: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to their experts and aggregate specialised outputs."""

        batch_size, seq_len, _ = pre_adapt.shape
        device = pre_adapt.device
        dtype = pre_adapt.dtype
        k = topk_indices.size(-1)

        total_tokens = batch_size * seq_len
        if total_tokens == 0:
            return torch.zeros(batch_size, seq_len, self.output_proj.out_features, device=device, dtype=dtype)

        # Flatten token positions and expert assignments
        token_positions = torch.arange(total_tokens, device=device, dtype=torch.long)
        token_positions = token_positions.unsqueeze(-1).expand(-1, k).reshape(-1)
        expert_indices = topk_indices.reshape(-1)
        weights = topk_weights.reshape(-1)

        # Gather adapter inputs for the dispatched tokens
        flat_pre = pre_adapt.reshape(total_tokens, self.adapt_dim)
        inputs = flat_pre.index_select(0, token_positions)  # [N, adapt_dim]

        # Select adapter weights/biases for the corresponding experts
        adapter_weight = self.adapter_weight.index_select(0, expert_indices).to(inputs.dtype)
        ln_weight = self.adapter_ln_weight.index_select(0, expert_indices).to(inputs.dtype)
        ln_bias = self.adapter_ln_bias.index_select(0, expert_indices).to(inputs.dtype)

        # Linear adapter + normalisation per dispatched token (batched)
        adapted = torch.bmm(inputs.unsqueeze(1), adapter_weight.transpose(1, 2)).squeeze(1)
        mean = adapted.mean(dim=-1, keepdim=True)
        var = adapted.var(dim=-1, unbiased=False, keepdim=True)
        adapted = (adapted - mean) / torch.sqrt(var + self.norm_eps)
        adapted = adapted * ln_weight + ln_bias

        # Shared projections back to model dimension
        hidden = self.expert_proj(adapted)
        del adapted  # Free memory immediately
        specialised = self.output_proj(hidden)
        output_dim = specialised.size(-1)
        del hidden  # Free memory immediately

        # Weight by router assignment and scatter back to token positions
        weighted = specialised * weights.to(specialised.dtype).unsqueeze(-1)
        del specialised  # Free memory immediately

        mixed = torch.zeros(total_tokens, output_dim, device=weighted.device, dtype=weighted.dtype)
        mixed.index_add_(0, token_positions, weighted)
        del weighted  # Free memory immediately

        return mixed.reshape(batch_size, seq_len, -1)

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

        # Routing (top-k) and expert dispatch
        # CRITICAL: Detach indices to prevent gradient computation through routing
        topk_weights, topk_indices, router_loss = self.router(normalized)
        topk_indices = topk_indices.detach()  # Don't backprop through routing decisions

        shared_proj, pre_adapt = self.expert_group.compute_shared(normalized)
        specialised = self.expert_group.mix_experts(pre_adapt, topk_indices, topk_weights)

        expert_output = shared_proj + 0.1 * specialised.to(shared_proj.dtype)

        # Add scaled residual connection
        output = residual + self.residual_scale * expert_output.to(residual.dtype)

        # Detach router_loss to prevent it from keeping the computation graph
        return output, router_loss.detach() 
