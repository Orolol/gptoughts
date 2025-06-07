"""Masked Diffusion Model (MDM) implementation."""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from ..blocks.mdm_block import MDMBlock
from ..blocks.normalization import RMSNorm
from .diffusion import MDMDiffusion


class MDMModel(nn.Module):
    """
    Masked Diffusion Model for language modeling.
    
    Uses a transformer encoder architecture with bidirectional attention
    and learns to denoise masked tokens through a diffusion process.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings (including mask token)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Position embeddings (learned, not RoPE since it's handled in attention)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        
        # Dropout
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            MDMBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer norm
        self.ln_f = RMSNorm(config.n_embd)
        
        # Output projection
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights between token embedding and output projection
        self.token_embedding.weight = self.lm_head.weight
        
        # Diffusion process
        self.diffusion = MDMDiffusion(config)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('o_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        print(f"MDM model initialized with {self.get_num_params()/1e6:.2f}M parameters")
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embedding.weight.numel()
        return n_params
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        masking_ratio: Optional[float] = None,
        return_dict: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass through MDM.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            labels: Target token IDs for training [batch_size, seq_len]
            timesteps: Diffusion timesteps [batch_size]
            masking_ratio: Optional fixed masking ratio
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary with 'logits', 'loss', 'masked_ids', 'mask'
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # Training mode: apply forward diffusion process
        if labels is not None:
            masked_ids, mask = self.diffusion.forward_process(
                labels, timesteps, masking_ratio
            )
            # Use masked version as input
            input_ids = masked_ids
        else:
            # Inference mode: input should already contain mask tokens
            mask = (input_ids == self.config.mask_token_id).float()
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        x = self.drop(token_embeds + position_embeds)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Get logits
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = self.diffusion.compute_loss(logits, labels, mask)
        
        if return_dict:
            return {
                'logits': logits,
                'loss': loss,
                'masked_ids': input_ids,
                'mask': mask
            }
        else:
            return (logits, loss, input_ids, mask)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate text using the reverse diffusion process.
        
        Args:
            input_ids: Optional partial sequence with mask tokens
            max_length: Maximum sequence length to generate
            num_steps: Number of denoising steps
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p filtering
            attention_mask: Optional attention mask
            
        Returns:
            Generated token IDs [batch_size, seq_len]
        """
        device = next(self.parameters()).device
        
        if input_ids is None:
            # Start with all mask tokens
            if max_length is None:
                max_length = self.config.block_size
            batch_size = 1
            input_ids = torch.full(
                (batch_size, max_length), 
                self.config.mask_token_id, 
                device=device
            )
        
        # Identify positions to generate (mask token positions)
        mask = (input_ids == self.config.mask_token_id).float()
        
        # Use diffusion sampling
        generated_ids = self.diffusion.sample(
            model=lambda x: self.forward(x, attention_mask=attention_mask)['logits'],
            masked_ids=input_ids,
            mask=mask,
            num_steps=num_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        return generated_ids
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Configure optimizer with weight decay."""
        # Start with all parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # Create optimizer
        use_fused = device_type == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        
        return optimizer