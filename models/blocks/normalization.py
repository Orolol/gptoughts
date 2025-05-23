"""Normalization layers for transformer models."""

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle Float8 or other exotic types that might cause issues
        try:
            # Always calculate norm in fp32 for stability, then convert back
            norm_x = self._norm(x.float())
            # Safely convert back to original type
            if x.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                norm_x = norm_x.to(x.dtype)
            return self.weight * norm_x
        except Exception as e:
            # Fallback to a safer implementation if there are dtype issues
            print(f"Warning: Using fallback normalization due to: {e}")
            working_type = torch.float32
            x_float = x.to(working_type)
            norm_x = self._norm(x_float)
            result = self.weight.to(working_type) * norm_x
            return result