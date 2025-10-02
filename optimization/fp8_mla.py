"""
FP8 optimizations for MLA models based on DeepSeek-V3's approach.

This module implements fine-grained quantization, high-precision accumulation,
and mixed precision strategies specifically tailored for MLA architecture.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Optional, Tuple, Union
import math


class FP8Quantizer:
    """
    Fine-grained quantization for FP8 training based on DeepSeek's approach.
    
    Features:
    - Tile-wise grouping for activations (1xNc elements)
    - Block-wise grouping for weights (NcxNc elements)
    - Online quantization with immediate scaling
    - E4M3 format preference for better precision
    """
    
    def __init__(self, tile_size: int = 128):
        """
        Args:
            tile_size: Size for tile/block quantization (default: 128 like DeepSeek)
        """
        self.tile_size = tile_size
        self.fp8_e4m3_max = 448.0  # Max value for E4M3 format
        self.fp8_e5m2_max = 57344.0  # Max value for E5M2 format
        
    def quantize_activation_tile(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize activations using tile-wise grouping (per token per tile_size channels).
        
        Args:
            x: Input tensor of shape [B, S, D] or [B, S, H, D]
            
        Returns:
            Tuple of (quantized_tensor, scale_factors)
        """
        original_shape = x.shape
        
        # Handle different input shapes
        if x.dim() == 3:  # [B, S, D]
            B, S, D = x.shape
            x_reshaped = x.view(B * S, -1, self.tile_size)
        elif x.dim() == 4:  # [B, S, H, D]
            B, S, H, D = x.shape
            x_reshaped = x.view(B * S * H, -1, self.tile_size)
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")
        
        # Calculate max values per tile
        max_vals = x_reshaped.abs().max(dim=-1, keepdim=True)[0]
        
        # Prevent division by zero
        max_vals = torch.clamp(max_vals, min=1e-12)
        
        # Calculate scale factors (online quantization)
        scale_factors = self.fp8_e4m3_max / max_vals
        
        # Quantize
        x_scaled = x_reshaped * scale_factors
        
        # Convert to FP8 E4M3 if available
        if hasattr(torch, 'float8_e4m3fn'):
            x_fp8 = x_scaled.to(torch.float8_e4m3fn)
        else:
            # Fallback: simulate FP8 with clipping
            x_fp8 = torch.clamp(x_scaled, -self.fp8_e4m3_max, self.fp8_e4m3_max)
        
        # Reshape back
        x_fp8 = x_fp8.view(original_shape)
        scale_factors = scale_factors.view(*original_shape[:-1], -1)
        
        return x_fp8, scale_factors
    
    def quantize_weight_block(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize weights using block-wise grouping (tile_size x tile_size blocks).
        
        Args:
            w: Weight tensor of shape [out_features, in_features]
            
        Returns:
            Tuple of (quantized_tensor, scale_factors)
        """
        out_features, in_features = w.shape
        
        # Pad if necessary
        out_pad = (self.tile_size - out_features % self.tile_size) % self.tile_size
        in_pad = (self.tile_size - in_features % self.tile_size) % self.tile_size
        
        if out_pad > 0 or in_pad > 0:
            w_padded = F.pad(w, (0, in_pad, 0, out_pad))
        else:
            w_padded = w
        
        # Reshape into blocks
        w_blocks = w_padded.view(
            -1, self.tile_size,
            w_padded.shape[1] // self.tile_size, self.tile_size
        ).permute(0, 2, 1, 3).contiguous()
        
        # Reshape for max calculation
        w_blocks = w_blocks.view(-1, self.tile_size * self.tile_size)
        
        # Calculate max values per block
        max_vals = w_blocks.abs().max(dim=-1, keepdim=True)[0]
        max_vals = torch.clamp(max_vals, min=1e-12)
        
        # Calculate scale factors
        scale_factors = self.fp8_e4m3_max / max_vals
        
        # Quantize
        w_scaled = w_blocks * scale_factors
        
        # Convert to FP8
        if hasattr(torch, 'float8_e4m3fn'):
            w_fp8 = w_scaled.to(torch.float8_e4m3fn)
        else:
            w_fp8 = torch.clamp(w_scaled, -self.fp8_e4m3_max, self.fp8_e4m3_max)
        
        # Reshape back (keeping padding for now)
        w_fp8 = w_fp8.view(-1, w_padded.shape[1] // self.tile_size, 
                          self.tile_size, self.tile_size).permute(0, 2, 1, 3)
        w_fp8 = w_fp8.contiguous().view(w_padded.shape)
        
        # Remove padding
        if out_pad > 0 or in_pad > 0:
            w_fp8 = w_fp8[:out_features, :in_features]
        
        # Reshape scale factors
        scale_factors = scale_factors.view(
            -1, w_padded.shape[1] // self.tile_size
        )
        
        return w_fp8, scale_factors
    
    def dequantize(self, x_fp8: torch.Tensor, scale_factors: torch.Tensor) -> torch.Tensor:
        """
        Dequantize FP8 tensor back to higher precision.
        
        Args:
            x_fp8: Quantized tensor
            scale_factors: Scale factors used for quantization
            
        Returns:
            Dequantized tensor in BF16 or FP32
        """
        # Convert from FP8 to higher precision
        if x_fp8.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            x_high = x_fp8.to(torch.bfloat16)
        else:
            x_high = x_fp8
        
        # Apply inverse scaling
        if scale_factors.shape != x_high.shape:
            # Handle tile-wise scale factors
            if x_high.dim() == 3 and scale_factors.dim() == 3:
                # Expand scale factors to match activation shape
                scale_factors = scale_factors.repeat_interleave(self.tile_size, dim=-1)
                scale_factors = scale_factors[..., :x_high.shape[-1]]
        
        x_dequant = x_high / scale_factors
        
        return x_dequant


class FP8LinearMLA(nn.Module):
    """
    FP8-optimized Linear layer for MLA with DeepSeek's improvements.
    
    Features:
    - Fine-grained quantization for activations and weights
    - High-precision accumulation using CUDA cores
    - Mixed precision computation (FP8 compute, higher precision storage)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 tile_size: int = 128, accumulation_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tile_size = tile_size
        self.accumulation_dtype = accumulation_dtype
        
        # Store weights in higher precision (BF16/FP32)
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.bfloat16))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter('bias', None)
        
        # Initialize quantizer
        self.quantizer = FP8Quantizer(tile_size)
        
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize using torch default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward_fp8(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using FP8 quantization with high-precision accumulation.
        """
        # Quantize activations (tile-wise)
        x_fp8, x_scale = self.quantizer.quantize_activation_tile(x)
        
        # Quantize weights (block-wise) 
        w_fp8, w_scale = self.quantizer.quantize_weight_block(self.weight)
        
        # Perform matrix multiplication with high-precision accumulation
        # This simulates the promotion to CUDA cores every Nc elements
        if hasattr(torch, 'float8_e4m3fn') and x_fp8.dtype == torch.float8_e4m3fn:
            # Use custom GEMM with periodic accumulation to FP32
            output = self._fp8_gemm_with_accumulation(x_fp8, w_fp8, x_scale, w_scale)
        else:
            # Fallback: dequantize and use standard matmul
            x_dequant = self.quantizer.dequantize(x_fp8, x_scale)
            w_dequant = self.quantizer.dequantize(w_fp8, w_scale)
            output = F.linear(x_dequant, w_dequant, self.bias)
        
        return output
    
    def _fp8_gemm_with_accumulation(self, x_fp8: torch.Tensor, w_fp8: torch.Tensor,
                                   x_scale: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
        """
        Simulate high-precision accumulation by breaking computation into chunks.
        
        This emulates DeepSeek's approach of promoting to CUDA cores every Nc elements
        for FP32 accumulation.
        """
        # For now, we'll use a simplified approach
        # In practice, this would require custom CUDA kernels
        
        # Convert to higher precision for accumulation
        x_acc = x_fp8.to(self.accumulation_dtype)
        w_acc = w_fp8.to(self.accumulation_dtype).t()
        
        # Apply scales during computation
        # This is more efficient than dequantizing first
        output = torch.matmul(x_acc, w_acc)
        
        # Apply combined scaling
        # Need to properly broadcast scale factors
        if x_scale.dim() == 3:  # [B, S, scale_dims]
            combined_scale = x_scale.unsqueeze(-1) * w_scale.unsqueeze(0).unsqueeze(0)
        else:
            combined_scale = x_scale * w_scale
        
        output = output / combined_scale.mean(dim=-2, keepdim=True)
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
        
        # Convert back to computation dtype
        output = output.to(torch.bfloat16)
        
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass - can switch between FP8 and regular."""
        if self.training and torch.cuda.is_available():
            # Use FP8 during training on CUDA
            return self.forward_fp8(x)
        else:
            # Use standard precision for inference or CPU
            return F.linear(x, self.weight, self.bias)


@contextmanager
def mla_fp8_mixed_precision(model: nn.Module, 
                           use_fp8_mla_params: bool = False,
                           optimize_memory: bool = True):
    """
    Context manager for mixed precision training of MLA models following DeepSeek's approach.
    
    Key principles:
    - Most compute operations in FP8
    - Critical components in higher precision (embeddings, normalization, attention)
    - Master weights and optimizer states in FP32/BF16
    - Low-precision communication for MoE models
    
    Args:
        model: The MLA model
        use_fp8_mla_params: Whether to use FP8 for MLA-specific parameters
        optimize_memory: Whether to optimize memory usage with low-precision storage
    """
    # Set environment variables for transformer engine
    os.environ["NVTE_ALLOW_FP8_USAGE"] = "1"
    os.environ["NVTE_FP8_RECIPE_FORMAT"] = "E4M3"  # Prefer E4M3 over HYBRID
    os.environ["NVTE_FP8_MHA"] = "0"  # Keep attention in higher precision
    os.environ["NVTE_FP8_FFN"] = "1"
    
    # Store original dtypes
    param_dtypes = {}
    
    try:
        # Configure model layers for mixed precision
        for name, module in model.named_modules():
            # Keep these in higher precision (following DeepSeek)
            if any(keep_type in name.lower() for keep_type in 
                   ['embed', 'norm', 'head', 'rope', 'attention']):
                continue
            
            # Convert linear layers to FP8
            if isinstance(module, nn.Linear):
                # Skip if it's MLA-specific and we want to keep those in FP16
                if not use_fp8_mla_params and 'mla' in name.lower():
                    continue
                
                # Store original dtype
                param_dtypes[name] = module.weight.dtype
                
                # For actual implementation, we'd replace with FP8LinearMLA
                # For now, just track what would be converted
                if hasattr(module, 'weight'):
                    # Would convert to FP8 computation here
                    pass
        
        yield
        
    finally:
        # Restore original dtypes if needed
        pass


def optimize_mla_fp8_training(model: nn.Module, config: dict) -> nn.Module:
    """
    Optimize MLA model for FP8 training following DeepSeek's approach.
    
    Args:
        model: MLA model to optimize
        config: Model configuration
        
    Returns:
        Optimized model
    """
    # Replace linear layers with FP8-optimized versions
    def replace_with_fp8_linear(module: nn.Module, name: str):
        for attr_name, child in module.named_children():
            full_name = f"{name}.{attr_name}" if name else attr_name
            
            if isinstance(child, nn.Linear):
                # Check if this should be kept in higher precision
                if any(keep in full_name.lower() for keep in 
                       ['embed', 'norm', 'head', 'rope']):
                    continue
                
                # Create FP8 linear replacement
                fp8_linear = FP8LinearMLA(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None
                )
                
                # Copy weights
                with torch.no_grad():
                    fp8_linear.weight.copy_(child.weight)
                    if child.bias is not None:
                        fp8_linear.bias.copy_(child.bias)
                
                # Replace module
                setattr(module, attr_name, fp8_linear)
            else:
                # Recurse
                replace_with_fp8_linear(child, full_name)
    
    # Apply replacements
    replace_with_fp8_linear(model, "")
    
    return model


def create_fp8_optimizer(model: nn.Module, 
                        optimizer_class: type,
                        lr: float,
                        use_low_precision_master_weights: bool = False,
                        **kwargs) -> torch.optim.Optimizer:
    """
    Create optimizer with FP8 training support.
    
    Following DeepSeek:
    - Stores first/second moments in BF16 instead of FP32
    - Master weights and gradients stay in FP32 for stability
    """
    # Separate parameters by precision requirements
    high_precision_params = []
    low_precision_params = []
    
    for name, param in model.named_parameters():
        if any(keep in name.lower() for keep in ['embed', 'norm', 'head']):
            high_precision_params.append(param)
        else:
            low_precision_params.append(param)
    
    # Create parameter groups with different settings
    param_groups = [
        {'params': high_precision_params, 'lr': lr},
        {'params': low_precision_params, 'lr': lr}
    ]
    
    # Add any additional kwargs
    for group in param_groups:
        group.update(kwargs)
    
    # Create optimizer
    optimizer = optimizer_class(param_groups)
    
    # Configure for low-precision moments if requested
    if use_low_precision_master_weights and hasattr(optimizer, 'state'):
        # This would require custom optimizer implementation
        # For now, just document the approach
        pass
    
    return optimizer