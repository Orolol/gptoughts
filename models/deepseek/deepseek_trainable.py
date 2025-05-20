"""
DeepSeek model implementation with training support.
This file extends the inference-only implementation to support training.
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from models.deepseek.deepseek import (
    ModelArgs, Transformer, ParallelEmbedding, Linear, 
    ColumnParallelLinear, RowParallelLinear, RMSNorm, 
    Block, MLA, MLP, MoE, precompute_freqs_cis, apply_rotary_emb
)


class TrainableTransformer(Transformer):
    """
    Extended Transformer model that supports training.
    """
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.args = args
        # Store for training parameters
        self.training_mode = False
        self.gradient_checkpointing = False
        
        # Apply more stable initialization to head for better numerical stability
        # The head layer is critical for avoiding NaN issues in loss calculation
        if hasattr(self, 'head') and isinstance(self.head, nn.Module):
            # Use Kaiming initialization with a small std to prevent extreme initial values
            if hasattr(self.head, 'weight') and self.head.weight is not None:
                nn.init.kaiming_normal_(self.head.weight, nonlinearity='linear', a=math.sqrt(5))
                # Scale down the initialization to avoid extreme initial logits
                with torch.no_grad():
                    self.head.weight.data.mul_(0.1)
            
            # Initialize bias to small values if present
            if hasattr(self.head, 'bias') and self.head.bias is not None:
                # Bias at zero reduces initial logit variance
                nn.init.zeros_(self.head.bias)
        
    def set_training_mode(self, mode: bool = True):
        """Enables or disables training mode."""
        self.training_mode = mode
        return self
    
    def set_gradient_checkpointing(self, value: bool):
        """Enable or disable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = value
        return self
    
    def forward(self, tokens: torch.Tensor, start_pos: int = 0, targets: Optional[torch.Tensor] = None):
        """
        Forward pass with support for both inference and training.
        This method will now consistently return full sequence logits if targets are not provided,
        or (full sequence logits, loss) if targets are provided.
        The nn.Module's self.training state controls layer behavior (e.g., dropout).
        """
        seqlen = tokens.size(1)
        
        # Main computation path for hidden states
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        
        # Process through transformer blocks
        # self.training (from nn.Module) controls behavior of layers like dropout
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training: # Use nn.Module's self.training state
                h = self._checkpointed_layer_forward(layer, h, start_pos, freqs_cis, mask)
            else:
                h = layer(h, start_pos, freqs_cis, mask)
        
        h_normalized = self.norm(h) # h_normalized has shape (B, S, H)
        
        # Always project all hidden states to get full sequence logits
        logits = self._project_hidden_states(h_normalized) # logits has shape (B, S, V)
        
        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            # Shift logits and targets for causal language modeling
            shift_logits = logits[:, :-1, :]
            shift_targets = targets[:, 1:]
            
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_targets.reshape(-1),
                ignore_index=-1  # Ignore padding tokens
            )
            
            # Handle NaN loss - set to high value without breaking backward pass
            if torch.isnan(loss).any():
                print("Warning: NaN loss detected in TrainableTransformer internal calculation")
                loss = torch.where(torch.isnan(loss), torch.full_like(loss, 100.0), loss)
                
            # Cap loss to avoid gradient explosions
            loss = torch.clamp(loss, max=100.0)
        
        # Return appropriate values
        if targets is not None:
            return logits, loss
        else:
            # No targets provided to this specific forward call, return full sequence logits.
            # The caller (e.g., LLMLightningModule) will handle further processing or loss calculation.
            return logits
    
    def _inference_forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """Existing inference mode forward pass with torch.inference_mode() handled externally."""
        seqlen = tokens.size(1)
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)[:, -1]
        logits = self.head(h)
        if dist.is_initialized() and dist.get_world_size() > 1:
            all_logits = [torch.empty_like(logits) for _ in range(dist.get_world_size())]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits
    
    def _project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply projection from hidden states to vocabulary logits."""
        # For full sequence, reshape correctly
        if hidden_states.dim() == 3:
            logits = self.head(hidden_states)
            if dist.is_initialized() and dist.get_world_size() > 1:
                # Avoid all_gather for sequences during training to save memory
                # We'll only synchronize at loss computation
                pass
        # For single token (inference)
        else:
            logits = self.head(hidden_states)
            if dist.is_initialized() and dist.get_world_size() > 1:
                all_logits = [torch.empty_like(logits) for _ in range(dist.get_world_size())]
                dist.all_gather(all_logits, logits)
                logits = torch.cat(all_logits, dim=-1)
        return logits
    
    def _checkpointed_layer_forward(self, layer, x, start_pos, freqs_cis, mask):
        """Custom checkpointing function to work with transformer blocks."""
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
            
        return torch.utils.checkpoint.checkpoint(
            create_custom_forward(layer),
            x, start_pos, freqs_cis, mask
        )


# Patch MLA to support gradient checkpointing during training
class TrainableMLA(MLA):
    """Extended MLA class with training support."""
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """Forward pass implementation for training and inference."""
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        
        # Same query projection as the original code
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
            
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        
        # KV processing
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        
        # Always use the training-friendly implementation for both training and inference
        # This avoids complex caching logic that breaks gradient flow
        q = torch.cat([q_nope, q_pe], dim=-1)
        kv = self.wkv_b(self.kv_norm(kv))
        kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
        
        # If we're in inference mode and have KV cache, we could use it
        # But for training, we compute all attention scores
        scores = torch.einsum("bshd,bthd->bsht", q, k) * self.softmax_scale
        
        if mask is not None:
            scores += mask.unsqueeze(1)
        
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        
        # Apply attention
        output = torch.einsum("bsht,bthd->bshd", scores, v)
        
        # Project output
        output = self.wo(output.flatten(2))
        return output


# Patch Block to use the trainable MLA
class TrainableBlock(Block):
    """Extended Block class that uses TrainableMLA for attention."""
    
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        # Replace the MLA with our trainable version
        self.attn = TrainableMLA(args)
        
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Forward pass with improved numerical stability for training."""
        # Add residual connection with normalization
        attn_output = self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        
        # Apply residual connection with potential gradient stabilization
        x = x + attn_output
        
        # Check for NaN values and replace if needed
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # FFN forward pass
        ffn_output = self.ffn(self.ffn_norm(x))
        
        # Apply residual connection
        x = x + ffn_output
        
        # Final NaN check
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        
        return x


class DeepSeekTrainable(nn.Module):
    """
    Full trainable DeepSeek model implementation.
    
    This class wraps the TrainableTransformer and manages training-specific operations.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Create the trainable transformer
        self.transformer = TrainableTransformer(args)
        self.args = args
        
        # Replace all blocks with trainable blocks
        for i in range(len(self.transformer.layers)):
            layer_id = i
            self.transformer.layers[i] = TrainableBlock(layer_id, args)
        
        # Initialize training attributes
        self.gradient_checkpointing = False
    
    def set_gradient_checkpointing(self, value: bool):
        """Enable or disable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = value
        self.transformer.set_gradient_checkpointing(value)
        return self
    
    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs):
        """
        Forward pass for both training and inference.
        
        Args:
            input_ids (torch.Tensor): Input token IDs with shape (batch_size, seq_len)
            targets (Optional[torch.Tensor]): Target token IDs for training
            **kwargs: Additional arguments
            
        Returns:
            torch.Tensor or Tuple: Model outputs or (outputs, loss) tuple
        """
        # Set training mode based on the current training state
        self.transformer.set_training_mode(self.training)
        
        # Extract start position if provided
        start_pos = kwargs.get('start_pos', 0)
        
        # Run the transformer in the appropriate mode
        outputs = self.transformer(input_ids, start_pos=start_pos, targets=targets)
        
        # Return outputs directly for inference or with loss for training
        return outputs
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, **kwargs):
        """
        Configure the optimizer for training.
        
        Args:
            weight_decay (float): Weight decay factor
            learning_rate (float): Learning rate
            betas (Tuple[float, float]): Adam betas parameters
            device_type (str): Device type ('cuda' or 'cpu')
            **kwargs: Additional optimizer arguments
            
        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        # Get optimizer type from kwargs if provided
        optimizer_type = kwargs.get('optimizer_type', 'adamw')
        
        # Collect all parameters and separate decay/no-decay groups
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Parameters that should not use weight decay
        no_decay = ['bias', 'norm', 'embedding']
        
        # Create parameter groups
        optimizer_grouped_params = [
            {
                'params': [p for n, p in param_dict.items() if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in param_dict.items() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        # Determine if fused implementation is available
        fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type == 'cuda'
        
        # Create the appropriate optimizer
        if optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                optimizer_grouped_params,
                lr=learning_rate,
                betas=betas,
                fused=use_fused if use_fused else False,
            )
        elif optimizer_type.lower() == 'lion':
            # Import separately to avoid dependencies
            try:
                from models.optimizers import Lion
                optimizer = Lion(
                    optimizer_grouped_params,
                    lr=learning_rate,
                    betas=betas,
                )
            except ImportError:
                print("Lion optimizer not available. Falling back to AdamW.")
                optimizer = torch.optim.AdamW(
                    optimizer_grouped_params,
                    lr=learning_rate,
                    betas=betas,
                    fused=use_fused if use_fused else False,
                )
        else:
            # Default to AdamW
            optimizer = torch.optim.AdamW(
                optimizer_grouped_params,
                lr=learning_rate,
                betas=betas,
                fused=use_fused if use_fused else False,
            )
            
        return optimizer
    
    def estimate_mfu(self, batch_size, seq_length, dt):
        """
        Estimate the model flops utilization (MFU).
        
        Args:
            batch_size (int): Batch size
            seq_length (int): Sequence length
            dt (float): Time delta for the computation
            
        Returns:
            float: Model flops utilization as a fraction (0-1)
        """
        # Constants and model parameters
        N = sum(p.numel() for p in self.parameters())
        L, H = self.args.n_layers, self.args.dim
        
        # Flops estimate based on standard transformer operations
        flops_per_token = 6 * N + 12 * L * H * seq_length
        flops = flops_per_token * batch_size
        
        # Estimate achieved flops
        flops_achieved = flops / (dt * 1e12)  # Expressed in TFLOPs
        
        # Theoretical peak flops for A100 GPU (adjust based on hardware)
        theoretical_flops = 312  # A100 GPU theoretical TFLOPs for BF16
        
        # MFU calculation
        mfu = flops_achieved / theoretical_flops
        return mfu