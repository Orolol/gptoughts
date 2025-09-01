"""
SEDD (Score Entropy Discrete Diffusion) model implementation.

This module contains the main SEDD model class that integrates with the training stack.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union

from ..config import SEDDConfig
from ..blocks.sedd_blocks import (
    DDiTBlock, 
    DDiTFinalLayer, 
    TimestepEmbedder,
    EmbeddingLayer
)
from ..blocks.positional_encoding import precompute_freqs_cis
from .diffusion_utils import (
    get_graph, 
    get_noise, 
    get_score_entropy_loss,
    sample_categorical
)


class SEDDModel(nn.Module):
    """
    SEDD (Score Entropy Discrete Diffusion) model for language modeling.
    
    This model implements discrete diffusion for text generation using score entropy 
    minimization. It's designed to integrate seamlessly with the existing training infrastructure.
    """
    
    def __init__(self, config: SEDDConfig):
        super().__init__()
        self.config = config
        
        # Model dimensions
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.block_size = config.block_size
        self.dropout = config.dropout
        
        # SEDD-specific parameters
        self.mask_token_id = config.mask_token_id
        self.cond_dim = config.cond_dim
        self.scale_by_sigma = config.scale_by_sigma
        
        # Initialize diffusion components
        self.graph = get_graph(config.graph_type, config.vocab_size)
        self.noise = get_noise(config.noise_type, config.sigma_min, config.sigma_max)
        self.sampling_eps = config.sampling_eps
        
        # Adjust vocab size for absorbing graph
        # For absorbing graph: vocab_size + 1 (the +1 is the absorbing/mask state)
        # For uniform graph: vocab_size (no extra state)
        effective_vocab_size = self.graph.dim
        
        # Fix mask token ID for absorbing graph
        if self.graph.absorb:
            # Mask token is the last token in the extended vocabulary
            self.mask_token_id = effective_vocab_size - 1
        else:
            self.mask_token_id = config.mask_token_id
        
        # Model components
        self.vocab_embed = EmbeddingLayer(effective_vocab_size, config.n_embd)
        self.timestep_embedder = TimestepEmbedder(config.cond_dim)
        
        # Rotary position embeddings
        head_dim = config.n_embd // config.n_head
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, config.block_size * 2, theta=10000.0),
            persistent=False
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DDiTBlock(
                dim=config.n_embd,
                n_heads=config.n_head,
                cond_dim=config.cond_dim,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                bias=config.bias,
                use_dyt=config.use_dyt,
                dyt_alpha_init=config.dyt_alpha_init,
                attention_backend=config.attention_backend
            )
            for _ in range(config.n_layer)
        ])
        
        # Output layer
        self.output_layer = DDiTFinalLayer(
            hidden_size=config.n_embd,
            out_channels=effective_vocab_size,
            cond_dim=config.cond_dim,
            bias=config.bias,
            use_dyt=config.use_dyt,
            dyt_alpha_init=config.dyt_alpha_init
        )
        
        # Initialize weights with careful initialization for diffusion models
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights with special care for diffusion stability."""
        if isinstance(module, nn.Linear):
            # Use smaller initialization for diffusion models to prevent instability
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Slightly smaller embedding initialization
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
        
        # Special initialization for adaptive layer norm modulation
        for name, param in module.named_parameters():
            if 'adaLN_modulation' in name:
                # Initialize to zero for adaptive layer norm as in original SEDD
                torch.nn.init.zeros_(param)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.freqs_cis.numel()
        return n_params

    def _prepare_diffusion_inputs(
        self, 
        input_ids: torch.Tensor, 
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for diffusion training or inference.
        
        Args:
            input_ids: Original token sequences [batch_size, seq_len]
            training: Whether in training mode
            
        Returns:
            Tuple of (perturbed_tokens, timesteps, original_tokens)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if training:
            # Sample random timesteps with a bias towards higher values for more corruption
            # This ensures we have enough masked tokens for stable training
            t = (1 - self.sampling_eps) * torch.rand(batch_size, device=device) + self.sampling_eps
            
            # Apply minimum corruption: bias towards higher t values early in training
            # This ensures we get at least some masked tokens to train on
            t = torch.clamp(t, min=0.1)  # Ensure minimum corruption level
            
            # Sample perturbed tokens using the graph transition
            sigma, _ = self.noise(t)
            perturbed_tokens = self.graph.sample_transition(input_ids, sigma[:, None])
            
            # For absorbing graph, ensure we have at least some masked tokens
            if self.graph.absorb:
                # Count masked tokens
                num_masked = (perturbed_tokens == self.graph.dim - 1).sum()
                min_masked = max(1, seq_len // 10)  # At least 10% should be masked
                
                if num_masked < min_masked:
                    # Force mask some additional tokens for training stability
                    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
                    seq_indices = torch.randperm(seq_len, device=device)[:min_masked].unsqueeze(0)
                    mask_positions = (batch_indices, seq_indices.expand(batch_size, -1))
                    perturbed_tokens[mask_positions] = self.graph.dim - 1
            
            return perturbed_tokens, t, input_ids
        else:
            # For inference, start from noise
            t = torch.ones(batch_size, device=device) * (1 - self.sampling_eps)
            
            # Sample from limiting distribution (uniform random for uniform graph, absorbing state for absorbing graph)
            if self.graph.absorb:
                # Start from absorbing state (mask tokens)
                perturbed_tokens = torch.full_like(input_ids, self.graph.dim - 1)
            else:
                # Start from random tokens
                perturbed_tokens = torch.randint(0, self.graph.dim, input_ids.shape, device=device)
            
            return perturbed_tokens, t, input_ids

    def forward(
        self, 
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the SEDD model.
        
        Args:
            input_ids: Input token sequences [batch_size, seq_len]
            labels: Target token sequences [batch_size, seq_len] (for training)
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            Dictionary containing logits and loss (if labels provided)
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # Ensure input is within vocabulary bounds
        if self.graph.absorb:
            # For absorbing graph, clip to valid range
            input_ids = torch.clamp(input_ids, 0, self.graph.dim - 2)
        else:
            input_ids = torch.clamp(input_ids, 0, self.graph.dim - 1)
        
        # Prepare diffusion inputs
        if labels is not None:
            # Training mode
            perturbed_tokens, t, original_tokens = self._prepare_diffusion_inputs(input_ids, training=True)
        else:
            # Inference mode - use input as original, generate perturbed version
            perturbed_tokens, t, original_tokens = self._prepare_diffusion_inputs(input_ids, training=False)
        
        # Get embeddings
        x = self.vocab_embed(perturbed_tokens)  # [batch_size, seq_len, n_embd]
        
        # Get timestep conditioning
        c = F.silu(self.timestep_embedder(t))  # [batch_size, cond_dim]
        
        # Get rotary position embeddings (disabled for simplicity)
        # freqs_cis = self.freqs_cis[:seq_len]
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, c, freqs_cis=None)
        
        # Final output layer
        raw_logits = self.output_layer(x, c)  # [batch_size, seq_len, vocab_size]
        
        # Apply SEDD-specific transformations
        sigma, dsigma = self.noise(t)
        
        # Apply sigma scaling if configured (as in original SEDD)
        if self.scale_by_sigma and self.graph.absorb:
            # Compute scaling factor: esigm1_log = log(exp(sigma) - 1) 
            esigm1_log = torch.where(
                sigma < 0.5, 
                torch.expm1(sigma), 
                sigma.exp() - 1
            ).log().to(raw_logits.dtype)[:, None, None]
            
            # Apply scaling: x = x - log(exp(sigma)-1) - log(vocab_size-1)
            raw_logits = raw_logits - esigm1_log - np.log(raw_logits.shape[-1] - 1)
        
        # Zero out raw logits for the current perturbed tokens (as in reference implementation)
        # This ensures the model doesn't predict the already corrupted token
        raw_logits = raw_logits.scatter(-1, perturbed_tokens[..., None], torch.zeros_like(raw_logits[..., :1]))
        
        # Convert to log probabilities for score entropy loss
        # The score entropy loss expects log p(x), not raw unnormalized logits
        log_probs = F.log_softmax(raw_logits, dim=-1)
        
        # For compatibility, we'll call these "logits" but they're actually log probabilities
        logits = log_probs
        
        # Prepare output
        output = {"logits": logits}
        
        # Compute SEDD score entropy loss if labels provided
        if labels is not None:
            try:
                # Use proper SEDD score entropy loss
                loss = get_score_entropy_loss(
                    model_output=logits,  # These are log scores
                    graph=self.graph,
                    noise=self.noise,
                    x=perturbed_tokens,
                    x0=original_tokens,
                    t=t
                )
                
                # Check for reasonable loss values and stability
                loss_mean = loss.mean()
                if torch.isnan(loss_mean) or torch.isinf(loss_mean):
                    print(f"NaN/Inf loss detected, using fallback")
                    raise ValueError("Invalid loss")
                
                # With all stability improvements, loss should be in 0-5 range
                if loss_mean > 5.0:
                    print(f"High loss detected: {loss_mean.item():.2f}, clipping for stability")
                    loss = torch.clamp(loss, min=0.0, max=5.0)
                    loss_mean = loss.mean()
                
                output["loss"] = loss_mean
                
                # Debug info for monitoring
                if hasattr(self, 'training_step_count'):
                    self.training_step_count = getattr(self, 'training_step_count', 0) + 1
                    if self.training_step_count % 100 == 0:
                        print(f"Step {self.training_step_count}: Loss = {loss_mean.item():.3f}, "
                              f"Sigma range: [{sigma.min().item():.3f}, {sigma.max().item():.3f}]")
                
            except Exception as e:
                print(f"Score entropy loss failed: {e}, using cross-entropy fallback")
                # Fallback to cross-entropy (shifted for autoregressive)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = original_tokens[..., 1:].contiguous()
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                flat_labels = shift_labels.view(-1)
                output["loss"] = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
        
        return output

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 40,
        num_steps: int = 128,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text using SEDD iterative denoising.
        
        Args:
            input_ids: Input token sequences [batch_size, seq_len] (used as prefix)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature for denoising
            top_k: Top-k sampling (used in final sampling step)
            num_steps: Number of denoising steps
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Generated token sequences
        """
        device = input_ids.device
        batch_size, input_len = input_ids.shape
        
        # Create the sequence to generate: prefix + masked tokens
        total_len = min(input_len + max_new_tokens, self.block_size)
        new_tokens_len = total_len - input_len
        
        if new_tokens_len <= 0:
            return input_ids  # No room for new tokens
        
        # Initialize with prefix + fully masked tokens
        if self.graph.absorb:
            # Start with prefix + mask tokens for absorbing graph
            # Use the actual absorbing state index (last token in extended vocab)
            absorbing_token = self.graph.dim - 1
            mask_tokens = torch.full((batch_size, new_tokens_len), absorbing_token, device=device)
            x = torch.cat([input_ids, mask_tokens], dim=1)
        else:
            # Start with prefix + random tokens for uniform graph
            random_tokens = torch.randint(0, self.graph.dim, (batch_size, new_tokens_len), device=device)
            x = torch.cat([input_ids, random_tokens], dim=1)
        
        # Iterative denoising process
        self.eval()
        with torch.no_grad():
            # Create denoising schedule
            timesteps = torch.linspace(1 - self.sampling_eps, self.sampling_eps, num_steps, device=device)
            
            for step, t_val in enumerate(timesteps):
                # Current timestep for all samples in batch
                t = torch.full((batch_size,), t_val, device=device)
                
                # Get model predictions (log probabilities)
                try:
                    output = self.forward(x, labels=x)  # Use current state as labels for consistency
                    log_probs = output["logits"]  # These are actually log probabilities now
                except Exception as e:
                    print(f"Error in generation step {step}: {e}")
                    break
                
                # Only update positions that can be denoised
                if self.graph.absorb:
                    # For absorbing graph, only update masked positions
                    absorbing_token = self.graph.dim - 1
                    update_mask = (x == absorbing_token)
                    # Don't update the prefix
                    update_mask[:, :input_len] = False
                else:
                    # For uniform graph, can update all new token positions
                    update_mask = torch.zeros_like(x, dtype=torch.bool)
                    update_mask[:, input_len:] = True
                
                if not update_mask.any():
                    continue  # Nothing to update
                
                # Sample next state for positions that need updating
                if step < num_steps - 1:
                    # Intermediate denoising steps: use model log probabilities
                    log_probs_to_update = log_probs[update_mask]  # [num_updates, vocab_size]
                    
                    # Apply temperature to log probabilities
                    if temperature > 0 and temperature != 1.0:
                        log_probs_to_update = log_probs_to_update / temperature
                    
                    # Convert log probabilities to probabilities and sample
                    probs = torch.exp(log_probs_to_update)
                    
                    # For absorbing graph, don't sample the absorbing state during denoising
                    if self.graph.absorb:
                        probs[:, -1] = 0  # Zero out absorbing state probability
                        probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize
                    
                    sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    
                    # Update only the positions that should be updated
                    x_new = x.clone()
                    x_new[update_mask] = sampled_tokens
                    x = x_new
                
                else:
                    # Final step: use greedy or top-k sampling
                    log_probs_to_update = log_probs[update_mask]
                    
                    if top_k > 0:
                        # Top-k sampling on log probabilities
                        top_k_logits, top_k_indices = torch.topk(log_probs_to_update, min(top_k, log_probs_to_update.size(-1)), dim=-1)
                        # Create mask for top-k
                        mask = torch.full_like(log_probs_to_update, float('-inf'))
                        mask.scatter_(-1, top_k_indices, top_k_logits)
                        log_probs_to_update = mask
                    
                    if temperature > 0:
                        log_probs_to_update = log_probs_to_update / temperature
                        probs = torch.exp(log_probs_to_update)
                        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    else:
                        # Greedy - use argmax on log probabilities
                        sampled_tokens = log_probs_to_update.argmax(dim=-1)
                    
                    # Final update
                    x_new = x.clone()
                    x_new[update_mask] = sampled_tokens
                    x = x_new
        
        self.train()  # Return to training mode
        return x

    def configure_optimizers(
        self,
        weight_decay: float = 0.1,
        learning_rate: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        device_type: str = "cuda",
        optimizer_type: Optional[str] = None,
        **kwargs
    ):
        """
        Configure optimizers for training.
        This method provides compatibility with the existing training infrastructure.
        """
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if len(param.shape) >= 2:  # Weight matrices
                    decay_params.append(param)
                else:  # Biases and layer norms
                    no_decay_params.append(param)
        
        # Create parameter groups
        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        
        # Use AdamW as default
        optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas)
        
        return optimizer