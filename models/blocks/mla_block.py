"""MLA Block components for transformer models."""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .normalization import RMSNorm
from .mla import MLA
from .mlp import MLP
from .tensor_utils import isolate_tensor, prevent_backward_reuse

class MLABlock(nn.Module):
    """
    Transformer block with Multi-Head Latent Attention (MLA) and MLP
    
    This block implements the architecture used in DeepSeek models with MLA for
    efficient memory usage through latent attention mechanisms.
    
    Attributes:
        layer_id (int): Identifier for this layer (used for debugging)
        attn_norm (nn.Module): Layer normalization for attention input
        attn (nn.Module): Multi-Head Latent Attention module
        ffn_norm (nn.Module): Layer normalization for feed-forward input
        ffn (nn.Module): MLP feed-forward network
        use_checkpoint (bool): Whether to use gradient checkpointing
    """
    def __init__(self, config, layer_id=None):
        super().__init__()
        # Layer identifier (for debugging)
        self.layer_id = layer_id
        
        # RMSNorm for attention and feed-forward
        self.attn_norm = RMSNorm(config.n_embd if hasattr(config, 'n_embd') else config.dim)
        self.ffn_norm = RMSNorm(config.n_embd if hasattr(config, 'n_embd') else config.dim)
        
        # MLA attention
        self.attn = MLA(config)
        
        # Always use standard MLP (no MoE)
        self.ffn = MLP(config)
        
        # Gradient checkpointing
        self.use_checkpoint = config.use_gradient_checkpointing if hasattr(config, 'use_gradient_checkpointing') else True

    def _attn_block(self, x, start_pos=0, freqs_cis=None, mask=None):
        """Attention portion of the block with appropriate normalization"""
        x_norm = self.attn_norm(x)
        return self.attn(x_norm, start_pos, freqs_cis, mask)

    def _ffn_block(self, x):
        """Feed-forward portion of the block with appropriate normalization"""
        # Always use regular MLP (no MoE)
        return self.ffn(self.ffn_norm(x))

    def forward(self, x, start_pos=0, freqs_cis=None, mask=None):
        """
        Forward pass for the MLABlock.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim)
            start_pos (int, optional): Starting position for attention caching. Defaults to 0.
            freqs_cis (torch.Tensor, optional): Precomputed position embeddings. Defaults to None.
            mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.
            
        Returns:
            torch.Tensor: Processed tensor with same shape as input
        """
        # Create a completely isolated copy of inputs to avoid double backward issues
        with torch.no_grad():
            x_local = x.detach().clone().requires_grad_(x.requires_grad)
            
            # Also isolate other tensor inputs
            if freqs_cis is not None:
                freqs_cis = freqs_cis.detach().clone()
            if mask is not None:
                mask = mask.detach().clone()
        
        # Disable gradient checkpointing when using compiled model
        if hasattr(self, '_compiled_forward'):
            self.use_checkpoint = False
        
        # Use the prevent_backward_reuse context manager for the forward pass
        with prevent_backward_reuse():
            if self.use_checkpoint and self.training:
                # Gradient checkpointing for memory efficiency
                def create_custom_forward(func):
                    def custom_forward(*inputs):
                        # Filter out None inputs
                        valid_inputs = [inp for inp in inputs if inp is not None]
                        return func(*valid_inputs)
                    return custom_forward

                # Attention with checkpoint
                attn_func = create_custom_forward(self._attn_block)
                attn_out = checkpoint.checkpoint(
                    attn_func, 
                    x_local, 
                    start_pos,
                    freqs_cis,
                    mask,
                    use_reentrant=False,
                    preserve_rng_state=True
                )
                # Create a fresh intermediate tensor
                x_mid = x_local.detach().clone().requires_grad_(True) + attn_out

                # FFN with checkpoint
                ffn_func = create_custom_forward(self._ffn_block)
                ffn_output = checkpoint.checkpoint(
                    ffn_func, 
                    x_mid,
                    use_reentrant=False,
                    preserve_rng_state=True
                )
                output = x_mid + ffn_output
            else:
                # Standard forward pass without checkpoint
                attn_out = self._attn_block(x_local, start_pos, freqs_cis, mask)
                # Create a fresh intermediate tensor
                x_mid = x_local + attn_out
                # Detach intermediate results to avoid double backward
                x_mid_detached = x_mid.detach().clone().requires_grad_(True)
                ffn_output = self._ffn_block(x_mid_detached)
                output = x_mid_detached + ffn_output

        # Create a fresh output tensor to completely isolate from computation graph
        with torch.no_grad():
            final_output = output.detach().clone().requires_grad_(True)
            # Manually copy the gradient function
            if output.requires_grad and final_output.requires_grad:
                final_output.retain_grad()
        
        return final_output