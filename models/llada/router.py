"""Router module for the LLaDA model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks.moe import Router

class LLaDARouter(Router):
    """
    Router module for LLaDA that extends the base Router with LLaDA-specific optimizations.
    """
    def __init__(self, input_dim: int, num_experts: int, k: int = 2, capacity_factor: float = 1.5):
        # Initialize the base Router
        super().__init__(input_dim, num_experts, k, capacity_factor)
        
        # Replace router implementation with a factorized version for better efficiency
        # Reduce parameter count by factorizing the routing network
        reduction_factor = 8
        hidden_dim = max(input_dim // reduction_factor, 32)
        
        # Remove the base router Sequential and replace with factorized version
        del self.router
        
        # Use separate layers for more efficient computation
        self.router_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.router_gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.router_norm = nn.LayerNorm(hidden_dim)
        
        # Adjust coefficients for LLaDA's specific needs
        self.router_z_loss_coef = 0.005  # Slightly reduced from base Router
        self.load_balance_coef = 0.005
        
        # Initialize weights for better gradient flow
        self._init_weights()
    
    def _init_weights(self):
        """Initialize router weights for better gradient flow"""
        nn.init.orthogonal_(self.router_projection.weight, gain=0.3)
        nn.init.orthogonal_(self.router_gate.weight, gain=0.3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        # Force cuBLAS to use Tensor Cores
        try:
            old_matmul_precision = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision('high')
        except AttributeError:
            old_matmul_precision = None
        
        # Compute router logits with memory optimization
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            try:
                # Use contiguous chunks for better memory layout
                if not x.is_contiguous():
                    x = x.contiguous()
                
                # Factorized routing computation (more efficient than base Router)
                x_reshaped = x.reshape(-1, self.input_dim)
                projected = self.router_projection(x_reshaped)
                normalized = self.router_norm(projected)
                router_logits = self.router_gate(normalized)
                
                # Scale logits by temperature
                temp_val = (self.temperature.abs() + 1e-6).item()
                router_logits = router_logits / temp_val
                
                # Compute routing probabilities
                routing_weights = F.softmax(router_logits, dim=-1)
                
                # Get top-k routing weights and indices
                effective_k = min(self.k, self.num_experts)
                top_k_weights, top_k_indices = torch.topk(routing_weights, effective_k, dim=-1)
                
                # Normalize top-k weights
                top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)
                
                # Create dispatch mask
                dispatch_mask = torch.zeros_like(routing_weights)
                dispatch_mask.scatter_(
                    dim=-1,
                    index=top_k_indices,
                    src=top_k_weights
                )
                
                # Compute load balancing loss
                ideal_load = torch.ones_like(routing_weights.mean(0)) / self.num_experts
                actual_load = routing_weights.mean(0)
                load_balance_loss = F.kl_div(
                    actual_load.log(),
                    ideal_load,
                    reduction='batchmean',
                    log_target=False
                )
                
                # Router z-loss
                router_z_loss = torch.square(router_logits).mean()
                
                # Combine losses
                router_loss = (
                    self.router_z_loss_coef * router_z_loss +
                    self.load_balance_coef * load_balance_loss
                )
                
                # Reshape dispatch mask for batch processing
                dispatch_mask = dispatch_mask.view(batch_size, seq_len, -1)
                
                # Reset precision if needed
                if old_matmul_precision is not None:
                    torch.set_float32_matmul_precision(old_matmul_precision)
                
                return routing_weights.detach(), dispatch_mask, router_loss
                
            except Exception as e:
                print(f"Router failed with error: {e}")
                # Fallback to uniform routing in case of error
                fallback_weights = torch.ones((batch_size * seq_len, self.num_experts), 
                                           device=x.device) / self.num_experts
                fallback_mask = fallback_weights.view(batch_size, seq_len, -1)
                fallback_loss = torch.tensor(0.0, device=x.device)
                
                # Reset precision if needed
                if old_matmul_precision is not None:
                    torch.set_float32_matmul_precision(old_matmul_precision)
                
                return fallback_weights, fallback_mask, fallback_loss 