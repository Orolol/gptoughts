"""LLaDA model implementation.

Large Language model with Diffusion and mAsking
Based on implementation described in https://arxiv.org/abs/2502.09992
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import gc

from ..config import LLaDAConfig
from .block import LLaDABlock

class LLaDAModel(nn.Module):
    """
    LLaDA model combining MoE architecture with diffusion-based language modeling.
    
    Key differences from standard language models:
    1. Uses bidirectional attention (not causal)
    2. Uses masking-based diffusion process instead of autoregressive generation
    3. Employs a low-confidence or random remasking strategy during generation
    """
    # Class-level warning counter
    _pos_warning_counter = 0
    _pos_warning_max = 5
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([LLaDABlock(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Output head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie token embedding weights with lm_head 
        self.lm_head.weight = self.tok_emb.weight
        
        # Dropout
        self.drop = nn.Dropout(config.dropout)
        
        # Special token ID for masking
        self.mask_token_id = config.mask_token_id
        
        # Gradient checkpointing flag
        self.use_checkpoint = getattr(config, 'use_checkpoint', True)
        
        # Apply weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with standard initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward_process(self, input_ids, eps=1e-3):
        """
        Apply random masking with varying ratios for diffusion process.
        
        This is the forward noise process from the diffusion framework.
        """
        batch_size, seq_len = input_ids.shape
        
        # Ensure mask_token_id is within vocabulary range
        safe_mask_token_id = min(self.mask_token_id, self.config.vocab_size - 1)
        
        # Sample random masking ratios between eps and 1-eps
        t = torch.rand(batch_size, device=input_ids.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, seq_len)
        
        # Apply masking randomly according to p_mask
        masked_indices = torch.rand((batch_size, seq_len), device=input_ids.device) < p_mask
        noisy_batch = torch.where(masked_indices, safe_mask_token_id, input_ids)
        
        return noisy_batch, masked_indices, p_mask

    def forward(self, input_ids, targets=None, apply_masking=True, eps=1e-3):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tensor of token indices [batch_size, seq_len]
            targets: Optional tensor of target indices [batch_size, seq_len]
            apply_masking: Whether to apply diffusion masking
            eps: Small epsilon value for numerical stability
            
        Returns:
            logits: Tensor of logits [batch_size, seq_len, vocab_size]
            loss: Optional cross-entropy loss
            router_loss: Optional aux loss from router
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # Apply masking if requested (for training)
        if apply_masking:
            noisy_batch, masked_indices, p_mask = self.forward_process(input_ids, eps)
        else:
            noisy_batch = input_ids
            masked_indices = None
            p_mask = None
        
        # Position embeddings
        pos = torch.arange(0, min(seq_len, self.config.block_size), device=device).unsqueeze(0)
        
        # Combine token embeddings and position embeddings
        x = self.tok_emb(noisy_batch)
        
        # Handle sequences longer than block_size
        if seq_len <= self.config.block_size:
            # Standard position embeddings
            x = x + self.pos_emb(pos)
        else:
            # For long sequences, pad position embeddings by repeating last one
            if LLaDAModel._pos_warning_counter < LLaDAModel._pos_warning_max:
                print(f"Warning: Sequence length {seq_len} exceeds block size {self.config.block_size}. Using position embeddings padding.")
                LLaDAModel._pos_warning_counter += 1
                
                if LLaDAModel._pos_warning_counter == LLaDAModel._pos_warning_max:
                    print(f"Note: Suppressing further position embedding warnings.")
            
            # Get position embeddings for available positions
            pos_emb_available = self.pos_emb(pos)
            
            # Create padded position embeddings with last position repeated
            pos_emb = torch.zeros((1, seq_len, self.config.n_embd), device=device)
            pos_emb[:, :self.config.block_size] = pos_emb_available
            pos_emb[:, self.config.block_size:] = pos_emb_available[:, -1:].expand(-1, seq_len - self.config.block_size, -1)
            
            # Add position embeddings to token embeddings
            x = x + pos_emb
        
        # Apply dropout
        x = self.drop(x)
        
        # Accumulate router loss
        total_router_loss = torch.tensor(0.0, device=device)
        
        # Forward pass through transformer blocks
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                # Use gradient checkpointing to save memory
                x, router_loss = checkpoint.checkpoint(block, x)
            else:
                # Standard forward pass
                x, router_loss = block(x)
                
            total_router_loss = total_router_loss + router_loss
        
        # Apply final layer norm
        x = self.ln_f(x)
        
        # Calculate logits
        logits = self.lm_head(x)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None and masked_indices is not None:
            # Calculate loss only on masked tokens
            masked_indices_flat = masked_indices.view(-1)
            if masked_indices_flat.any():
                # Extract values for masked positions only
                masked_logits = logits.view(-1, logits.size(-1))[masked_indices_flat]
                masked_targets = targets.view(-1)[masked_indices_flat]
                masked_p_mask = p_mask.view(-1)[masked_indices_flat]
                
                # Cross entropy weighted by masking probability
                token_loss = F.cross_entropy(
                    masked_logits,
                    masked_targets,
                    reduction='none'
                ) / masked_p_mask
                
                # Normalize loss
                loss = token_loss.sum() / (batch_size * seq_len)
            else:
                # No masked tokens to calculate loss on
                loss = torch.tensor(0.0, device=device)
        
        return logits, loss, total_router_loss
    
    @torch.no_grad()
    def generate(self, prompt, steps=32, gen_length=32, block_length=32, temperature=None, remasking='low_confidence'):
        """
        Generate text using LLaDA diffusion process.
        
        Args:
            prompt: Input token IDs [batch_size, prompt_length]
            steps: Number of diffusion steps
            gen_length: Length of text to generate
            block_length: Size of generation blocks for semi-autoregressive generation
            temperature: Temperature for sampling (None = use config value)
            remasking: Remasking strategy ('low_confidence' or 'random')
            
        Returns:
            generated: Generated token IDs [batch_size, prompt_length + gen_length]
        """
        device = prompt.device
        
        # Use values from config if not specified
        temperature = temperature if temperature is not None else self.config.temperature
        remasking = remasking if remasking is not None else self.config.remasking
        
        # Safety checks on parameters
        gen_length = max(1, min(gen_length, 1024))
        block_length = max(1, min(block_length, gen_length))
        steps = max(1, min(steps, 100))
        
        # Enable KV caching for faster generation
        self.enable_kv_cache()
        
        try:
            # Initialize sequence with prompt and masked tokens
            batch_size, prompt_length = prompt.shape
            safe_mask_token_id = min(self.mask_token_id, self.config.vocab_size - 1)
            
            # Create full sequence with masked generation region
            generated = torch.zeros((batch_size, prompt_length + gen_length), 
                                   dtype=torch.long, device=device)
            generated[:, :prompt_length] = prompt
            generated[:, prompt_length:] = safe_mask_token_id
            
            # Iterative demasking
            for step in range(steps):
                # Process in smaller blocks to save memory
                for block_start in range(0, gen_length, block_length):
                    block_end = min(block_start + block_length, gen_length)
                    current_length = prompt_length + block_end
                    
                    # Get predictions for current sequence
                    logits, _, _ = self.forward(generated[:, :current_length], apply_masking=False)
                    
                    # Get probabilities for the generation region
                    gen_logits = logits[:, prompt_length:current_length]
                    
                    # Apply temperature sampling
                    if temperature > 0:
                        gen_logits = gen_logits / max(temperature, 1e-6)
                    
                    # Convert to probabilities
                    probs = F.softmax(gen_logits, dim=-1)
                    
                    # Determine which tokens to update
                    unmask_frac = (step + 1) / steps  # Gradually unmask more tokens
                    
                    # Find masked positions in the generation region
                    mask_region = generated[:, prompt_length:current_length]
                    masked_positions = (mask_region == safe_mask_token_id)
                    
                    if masked_positions.any():
                        # Handle different remasking strategies
                        if remasking == 'low_confidence':
                            # Update tokens with highest model confidence
                            confidence = probs.max(dim=-1).values
                            
                            # For each sequence, find positions to update
                            for b in range(batch_size):
                                seq_masked = masked_positions[b]
                                if not seq_masked.any():
                                    continue
                                    
                                # Get confidence values for masked positions
                                seq_confidence = confidence[b][seq_masked]
                                
                                # Determine how many tokens to unmask
                                num_to_unmask = max(1, int(unmask_frac * seq_masked.sum()))
                                num_to_unmask = min(num_to_unmask, seq_masked.sum())
                                
                                # Find highest confidence positions
                                _, top_indices = seq_confidence.topk(num_to_unmask)
                                
                                # Get masked position indices
                                masked_indices = torch.nonzero(seq_masked).squeeze(-1)
                                unmask_positions = masked_indices[top_indices]
                                
                                # Sample from the model distribution
                                selected_probs = probs[b][unmask_positions]
                                sampled_tokens = torch.multinomial(selected_probs, 1).squeeze(-1)
                                
                                # Update the sequence
                                position_offset = prompt_length
                                generated[b, unmask_positions + position_offset] = sampled_tokens
                        
                        elif remasking == 'random':
                            # Randomly select masked positions to update
                            for b in range(batch_size):
                                seq_masked = masked_positions[b]
                                if not seq_masked.any():
                                    continue
                                
                                # Determine how many tokens to unmask
                                num_to_unmask = max(1, int(unmask_frac * seq_masked.sum()))
                                num_to_unmask = min(num_to_unmask, seq_masked.sum())
                                
                                # Get masked position indices
                                masked_indices = torch.nonzero(seq_masked).squeeze(-1)
                                
                                # Randomly select positions to unmask
                                perm = torch.randperm(masked_indices.size(0), device=device)
                                unmask_positions = masked_indices[perm[:num_to_unmask]]
                                
                                # Sample from the model distribution
                                selected_probs = probs[b][unmask_positions]
                                sampled_tokens = torch.multinomial(selected_probs, 1).squeeze(-1)
                                
                                # Update the sequence
                                position_offset = prompt_length
                                generated[b, unmask_positions + position_offset] = sampled_tokens
            
            # Handle any remaining masked tokens
            remaining_masked = (generated == safe_mask_token_id)
            if remaining_masked.any():
                # Final forward pass to handle any remaining tokens
                final_logits, _, _ = self.forward(generated, apply_masking=False)
                final_probs = F.softmax(final_logits, dim=-1)
                
                # For each masked position, sample a replacement
                masked_indices = torch.nonzero(remaining_masked)
                for b, pos in masked_indices:
                    sampled_token = torch.multinomial(final_probs[b, pos], 1).item()
                    generated[b, pos] = sampled_token
            
            # Reset KV cache to free memory
            self.reset_kv_cache()
            
            return generated
            
        except Exception as e:
            print(f"Error during generation: {e}")
            # Reset KV cache in case of error
            self.reset_kv_cache()
            return prompt
    
    def enable_kv_cache(self):
        """Enable KV caching for efficient generation"""
        for block in self.blocks:
            block.attn.kv_cache_enabled = True
            block.attn._cached_k = None
            block.attn._cached_v = None
    
    def disable_kv_cache(self):
        """Disable KV caching to free memory"""
        for block in self.blocks:
            block.attn.kv_cache_enabled = False
            block.attn._cached_k = None
            block.attn._cached_v = None
    
    def reset_kv_cache(self):
        """Clear KV cache to free memory"""
        for block in self.blocks:
            block.attn._cached_k = None
            block.attn._cached_v = None
        
        # Force CUDA memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def set_gradient_checkpointing(self, value):
        """Enable or disable gradient checkpointing"""
        self.use_checkpoint = value
        
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure optimizer with weight decay, specialized for LLaDA.
        
        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            betas: Adam beta parameters
            device_type: Device type ('cuda' or 'cpu')
            
        Returns:
            Configured optimizer
        """
        from ..optimizers import configure_optimizer_for_llada
        return configure_optimizer_for_llada(
            model=self,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
            device_type=device_type
        ) 