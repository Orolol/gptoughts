"""Diffusion utilities for Masked Diffusion Model."""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class MDMDiffusion:
    """
    Diffusion process for Masked Diffusion Model.
    Handles noise schedules, forward masking, and reverse denoising.
    """
    
    def __init__(self, config):
        self.config = config
        self.num_steps = config.diffusion_steps
        self.mask_token_id = config.mask_token_id
        
        # Create noise schedule
        self.betas = self._create_noise_schedule(
            config.noise_schedule,
            config.diffusion_steps,
            config.beta_start,
            config.beta_end
        )
        
        # Pre-compute diffusion parameters
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # For sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For posterior computation
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _create_noise_schedule(
        self, 
        schedule_type: str, 
        num_steps: int, 
        beta_start: float, 
        beta_end: float
    ) -> torch.Tensor:
        """Create beta schedule for diffusion process."""
        if schedule_type == "linear":
            return torch.linspace(beta_start, beta_end, num_steps)
        elif schedule_type == "cosine":
            # Cosine schedule from improved DDPM
            steps = torch.arange(num_steps + 1, dtype=torch.float32) / num_steps
            alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            return torch.clamp(betas, min=0.0001, max=0.999)
        elif schedule_type == "sqrt":
            return torch.sqrt(torch.linspace(beta_start**2, beta_end**2, num_steps))
        else:
            raise ValueError(f"Unknown noise schedule: {schedule_type}")
    
    def forward_process(
        self, 
        input_ids: torch.Tensor, 
        timesteps: Optional[torch.Tensor] = None,
        masking_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process - randomly mask tokens.
        
        Args:
            input_ids: Original token IDs [batch_size, seq_len]
            timesteps: Diffusion timesteps [batch_size]
            masking_ratio: Optional fixed masking ratio (overrides timestep-based)
            
        Returns:
            Tuple of (masked_ids, mask) where mask indicates masked positions
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if masking_ratio is None:
            # Use timestep-based masking ratio
            if timesteps is None:
                # Sample random timesteps
                timesteps = torch.randint(0, self.num_steps, (batch_size,), device=device)
            
            # Convert timesteps to masking ratios (more masking at higher timesteps)
            alphas_t = self.alphas_cumprod[timesteps].to(device)
            masking_ratios = 1.0 - alphas_t  # [batch_size]
        else:
            # Use fixed masking ratio for all samples
            masking_ratios = torch.full((batch_size,), masking_ratio, device=device)
        
        # Create random mask for each sequence
        mask = torch.zeros(batch_size, seq_len, device=device)
        
        for i in range(batch_size):
            # Number of tokens to mask
            num_mask = int(masking_ratios[i].item() * seq_len)
            
            # Randomly select positions to mask
            if num_mask > 0:
                mask_indices = torch.randperm(seq_len, device=device)[:num_mask]
                mask[i, mask_indices] = 1
        
        # Apply mask
        masked_ids = input_ids.clone()
        masked_ids[mask.bool()] = self.mask_token_id
        
        return masked_ids, mask
    
    def compute_loss(
        self,
        model_output: torch.Tensor,
        target_ids: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute diffusion loss for masked positions only.
        
        Args:
            model_output: Model predictions [batch_size, seq_len, vocab_size]
            target_ids: Original token IDs [batch_size, seq_len]
            mask: Binary mask indicating masked positions [batch_size, seq_len]
            
        Returns:
            Scalar loss value
        """
        # Only compute loss on masked positions
        masked_positions = mask.bool()
        
        if not masked_positions.any():
            return torch.tensor(0.0, device=model_output.device)
        
        # Flatten for loss computation
        logits_masked = model_output[masked_positions]  # [num_masked, vocab_size]
        targets_masked = target_ids[masked_positions]   # [num_masked]
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits_masked, targets_masked)
        
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        model,
        masked_ids: torch.Tensor,
        mask: torch.Tensor,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Reverse diffusion process - iteratively denoise masked tokens.
        
        Args:
            model: The MDM model
            masked_ids: Input with masked tokens [batch_size, seq_len]
            mask: Binary mask indicating positions to generate [batch_size, seq_len]
            num_steps: Number of denoising steps (default: use all)
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering
            
        Returns:
            Generated token IDs [batch_size, seq_len]
        """
        if num_steps is None:
            num_steps = self.num_steps
        
        device = masked_ids.device
        batch_size, seq_len = masked_ids.shape
        
        # Start with fully masked positions
        current_ids = masked_ids.clone()
        remaining_mask = mask.clone()
        
        # Greedy sampling following MaskGIT approach
        for step in range(num_steps):
            if not remaining_mask.any():
                break
                
            # Get model predictions
            logits = model(current_ids)  # [batch_size, seq_len, vocab_size]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Get probabilities for masked positions
            probs = F.softmax(logits, dim=-1)
            
            # Sample or take argmax
            if temperature > 0:
                # Sample from distribution
                if top_k is not None or top_p is not None:
                    probs = self._apply_top_k_top_p_filtering(logits, top_k, top_p)
                sampled_ids = torch.multinomial(
                    probs.view(-1, probs.size(-1)), 
                    num_samples=1
                ).view(batch_size, seq_len)
            else:
                # Greedy decoding
                sampled_ids = torch.argmax(logits, dim=-1)
            
            # Compute confidence scores for masked positions
            confidence = torch.gather(probs, -1, sampled_ids.unsqueeze(-1)).squeeze(-1)
            confidence = confidence * remaining_mask  # Only for currently masked positions
            
            # Determine how many tokens to unmask this step
            num_masked = remaining_mask.sum().item()
            num_to_unmask = max(1, num_masked // (num_steps - step))
            
            # Find least confident predictions
            if num_to_unmask < num_masked:
                # Flatten confidence scores for masked positions
                masked_confidence = confidence[remaining_mask.bool()]
                threshold = torch.kthvalue(masked_confidence, num_to_unmask)[0]
                
                # Unmask positions with confidence >= threshold
                unmask_positions = (confidence >= threshold) & remaining_mask.bool()
            else:
                # Unmask all remaining positions
                unmask_positions = remaining_mask.bool()
            
            # Update tokens
            current_ids[unmask_positions] = sampled_ids[unmask_positions]
            remaining_mask[unmask_positions] = 0
        
        # Final pass for any remaining masked tokens
        if remaining_mask.any():
            logits = model(current_ids)
            sampled_ids = torch.argmax(logits, dim=-1)
            current_ids[remaining_mask.bool()] = sampled_ids[remaining_mask.bool()]
        
        return current_ids
    
    def _apply_top_k_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """Apply top-k and/or top-p filtering to logits."""
        if top_k is not None:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        return F.softmax(logits, dim=-1)