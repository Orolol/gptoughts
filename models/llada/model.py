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
from train.train_utils import estimate_mfu as utils_estimate_mfu

# --- BD3-LM Helper Functions moved inside class ---

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
        """Initializes the LLaDAModel."""
        super().__init__()
        self.config = config
        
        # --- BD3-LM Specific Config ---
        # Add block_length for BD3 processing, default could be config.block_size or smaller
        # Use a reasonable default if not provided in config
        default_bd3_block = config.block_size // 4 if config.block_size >= 16 else 16
        self.bd3_block_length = getattr(config, 'bd3_block_length', default_bd3_block)
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # Increase position embedding size to handle longer sequences during generation
        max_pos_emb = max(config.block_size * 2, 2048)  # At least 2x block size or 2048
        self.pos_emb = nn.Embedding(max_pos_emb, config.n_embd)
        
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
            # Corrected duplicate line
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
 
    # --- BD3-LM Helper Methods ---

    def _create_bd3_attention_mask(self, seq_len, block_length, device):
        """
        Creates the specialized Mfull attention mask (boolean) for BD3-LM vectorized training.
        False indicates attention is allowed, True indicates it's masked (for SDPA).
        Output mask shape: [2 * seq_len, 2 * seq_len] (broadcastable).

        Structure (conceptual):
            Mfull = [ MBC     0   ]  <- Clean attends Clean (causal), Clean CANNOT attend Noisy
                    [ MOBC   MBD  ]  <- Noisy attends Clean (offset causal), Noisy attends Noisy (block-diag)
        Indices 0..L-1 are clean, L..2L-1 are noisy
        """
        L = seq_len
        L_prime = block_length
        full_len = 2 * L

        # Create indices for rows (query) and columns (key)
        q_indices = torch.arange(full_len, device=device)
        k_indices = torch.arange(full_len, device=device)

        # Calculate block index for each position
        # Block index for clean part (indices 0 to L-1)
        q_block_idx_clean = q_indices[:L] // L_prime
        k_block_idx_clean = k_indices[:L] // L_prime
        # Block index for noisy part (indices L to 2L-1)
        q_block_idx_noisy = (q_indices[L:] - L) // L_prime
        k_block_idx_noisy = (k_indices[L:] - L) // L_prime

        # Initialize full mask with True (mask everything)
        # Remember: True means MASKED in SDPA boolean masks
        mask = torch.ones((full_len, full_len), dtype=torch.bool, device=device)

        # --- Populate the four quadrants ---
        # Quadrant indices: [Query Range, Key Range]

        # 1. MBC (upper-left, [:L, :L]): Clean attends Clean (Block-causal MBC)
        # Allowed if q_block_idx_clean[q] >= k_block_idx_clean[k]
        q_block_upper_left = q_block_idx_clean
        k_block_upper_left = k_block_idx_clean
        mbc_mask = q_block_upper_left[:, None] < k_block_upper_left[None, :] # True if q_block < k_block (masked)
        mask[:L, :L] = mbc_mask

        # 2. Zero Mask (upper-right, [:L, L:]): Clean attends Noisy (Never allowed)
        # This quadrant remains True (masked) as initialized.
        # mask[:L, L:] = True

        # 3. MOBC (lower-left, [L:, :L]): Noisy attends Clean (Offset Block-causal)
        # Allowed if q_block_idx_noisy[q] > k_block_idx_clean[k]
        q_block_lower_left = q_block_idx_noisy
        k_block_lower_left = k_block_idx_clean
        # Mask if q_block <= k_block
        mobc_mask = q_block_lower_left[:, None] <= k_block_lower_left[None, :] # True if q_block <= k_block (masked)
        mask[L:, :L] = mobc_mask

        # 4. MBD (lower-right, [L:, L:]): Noisy attends Noisy (Block-diagonal)
        # Allowed if q_block_idx_noisy[q] == k_block_idx_noisy[k]
        q_block_lower_right = q_block_idx_noisy
        k_block_lower_right = k_block_idx_noisy
        # Mask if q_block != k_block
        mbd_mask = q_block_lower_right[:, None] != k_block_lower_right[None, :] # True if q_block != k_block (masked)
        mask[L:, L:] = mbd_mask

        # SDPA expects True where attention should be masked.
        # The mask is already constructed this way.
        # Return the (2L, 2L) mask. SDPA should broadcast it correctly.
        return mask

    def _bd3_noise_process(self, input_ids, block_length, beta=0.0, omega=1.0, eps=1e-6): # Added return for p_mask_rates
        """
        Applies block-wise noise based on BD3-LM principles using clipped noise schedule.
        Samples different noise levels t_b for each block by sampling masking rate from U[beta, omega].
        Uses masking as the noise process.
        
        Args:
            input_ids: The clean input token IDs.
            beta (float): Minimum masking rate (1 - alpha_t_max).
            omega (float): Maximum masking rate (1 - alpha_t_min).
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        num_blocks = (seq_len + block_length - 1) // block_length
        
        noisy_batch = input_ids.clone()
        
        # Ensure mask_token_id is valid
        safe_mask_token_id = min(self.mask_token_id, self.config.vocab_size - 1)

        all_masked_indices = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
        all_p_mask_rates = torch.zeros_like(input_ids, dtype=torch.float32, device=device) # Store per-token mask rate
        
        for b in range(num_blocks):
            start_idx = b * block_length
            end_idx = min((b + 1) * block_length, seq_len)
            current_block_len = end_idx - start_idx
            
            if current_block_len <= 0:
                continue

            # Sample masking rate p = 1 - alpha_t uniformly from [beta, omega]
            # Ensure beta and omega are valid
            beta_clipped = max(eps, beta)
            omega_clipped = min(1.0 - eps, omega)
            p_mask_rate_b = torch.rand(batch_size, 1, device=device) * (omega_clipped - beta_clipped) + beta_clipped

            # Create mask for the current block
            block_mask = torch.rand(batch_size, current_block_len, device=device) < p_mask_rate_b
            
            # Apply mask to the corresponding part of noisy_batch
            noisy_batch[:, start_idx:end_idx] = torch.where(
                block_mask, 
                torch.tensor(safe_mask_token_id, device=device, dtype=input_ids.dtype), # Ensure tensor type match
                noisy_batch[:, start_idx:end_idx]
            ) # Corrected parenthesis
            all_masked_indices[:, start_idx:end_idx] = block_mask
            all_p_mask_rates[:, start_idx:end_idx] = p_mask_rate_b.expand(-1, current_block_len) # Store the rate used

        # Note: The original LLaDA forward_process returned p_mask, which might be needed
        # for the weighted LBD loss later. For simple CE loss, we just need masked_indices.
        # Return the rates used for masking, needed for LBD loss.
        return noisy_batch, all_masked_indices, all_p_mask_rates

    # --- End BD3-LM Helper Methods ---

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

    def forward(self, input_ids, targets=None, apply_masking=True, eps=1e-3, use_bd3_training=False, past_key_values=None, return_kv_cache=False): # Added KV cache args
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tensor of token indices [batch_size, seq_len]
            targets: Optional tensor of target indices [batch_size, seq_len]
            apply_masking: Whether to apply diffusion masking (original LLaDA or BD3)
            eps: Small epsilon value for numerical stability
            use_bd3_training: Flag to enable BD3-LM vectorized training logic
            past_key_values: Optional list of tuples (k, v) for each layer (used in generation).
            return_kv_cache: If True, return the computed KV cache for all layers.
            
        Returns:
            logits: Tensor of logits [batch_size, seq_len, vocab_size] (or [batch_size, 2*seq_len, vocab_size] for BD3)
            loss: Optional cross-entropy loss
            router_loss: Optional aux loss from router
            present_key_values: Optional list of tuples (k, v) for each layer (if return_kv_cache is True).
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        attn_mask = None # Default: no mask (original LLaDA bidirectional)
        
        if use_bd3_training:
            # --- BD3-LM Vectorized Training Path ---
            assert targets is not None, "Targets must be provided for BD3 training"
            
            # 1. Apply block-wise noise using the class method
            # Now also returns p_mask_rates, though not strictly needed for simplified LBD loss
            noisy_batch, masked_indices, _ = self._bd3_noise_process(
                input_ids,
                self.bd3_block_length,
                eps
            )
            
            # 2. Concatenate clean and noisy inputs
            # Input shape becomes [batch_size, 2 * seq_len]
            combined_input_ids = torch.cat([input_ids, noisy_batch], dim=1)
            current_seq_len = 2 * seq_len
            
            # 3. Create the specialized BD3 attention mask using the class method
            # Mask shape needs to match attention mechanism [batch_size, num_heads, 2*seq_len, 2*seq_len] or broadcastable
            attn_mask = self._create_bd3_attention_mask(seq_len, self.bd3_block_length, device)
            
            # 4. Embeddings
            x = self.tok_emb(combined_input_ids)
            
            # 5. Position Embeddings for combined length
            # Position embeddings for combined length
            max_pos_len = self.pos_emb.weight.shape[0]
            if current_seq_len <= max_pos_len:
                # We have enough position embeddings
                pos = torch.arange(0, current_seq_len, device=device).unsqueeze(0)
                pos_emb_lookup = self.pos_emb(pos)
                x = x + pos_emb_lookup
            else:
                # Need to handle sequences longer than max position embeddings
                if LLaDAModel._pos_warning_counter < LLaDAModel._pos_warning_max:
                    print(f"Warning: BD3 combined length {current_seq_len} exceeds max pos embeddings {max_pos_len}. Using cyclic embeddings.")
                    LLaDAModel._pos_warning_counter += 1
                    if LLaDAModel._pos_warning_counter == LLaDAModel._pos_warning_max: print("Note: Suppressing further pos emb warnings.")
                
                # Use cyclic position embeddings for very long sequences
                pos_indices = torch.arange(0, current_seq_len, device=device) % max_pos_len
                pos_emb_lookup = self.pos_emb(pos_indices.unsqueeze(0))
                x = x + pos_emb_lookup
                
        else:
            # --- Original LLaDA Path ---
            current_seq_len = seq_len
            # Apply masking if requested (for training)
            if apply_masking:
                noisy_batch, masked_indices, p_mask = self.forward_process(input_ids, eps)
            else:
                noisy_batch = input_ids
                masked_indices = None
                p_mask = None # Needed for original loss calc
            
            # Embeddings
            x = self.tok_emb(noisy_batch)
            
            # Position embeddings
            max_pos_len = self.pos_emb.weight.shape[0]
            if current_seq_len <= max_pos_len:
                # Standard position embeddings
                pos = torch.arange(0, current_seq_len, device=device).unsqueeze(0)
                pos_emb_lookup = self.pos_emb(pos)
                x = x + pos_emb_lookup
            else:
                # Handle sequences longer than max position embeddings
                if LLaDAModel._pos_warning_counter < LLaDAModel._pos_warning_max:
                    print(f"Warning: Sequence length {current_seq_len} exceeds max pos embeddings {max_pos_len}. Using cyclic embeddings.")
                    LLaDAModel._pos_warning_counter += 1
                    if LLaDAModel._pos_warning_counter == LLaDAModel._pos_warning_max: print("Note: Suppressing further pos emb warnings.")
                
                # Use cyclic position embeddings
                pos_indices = torch.arange(0, current_seq_len, device=device) % max_pos_len
                pos_emb_lookup = self.pos_emb(pos_indices.unsqueeze(0))
                x = x + pos_emb_lookup
        
        # Apply dropout (common to both paths)
        x = self.drop(x)
        
        # Accumulate router loss
        total_router_loss = torch.tensor(0.0, device=device)
        
        # List to store KV cache for each layer if requested
        present_key_values_list = [] if return_kv_cache else None
        
        # Forward pass through transformer blocks (common, but uses attn_mask for BD3)
        for i, block in enumerate(self.blocks):
            # Pass the attn_mask (will be None for original LLaDA path)
            block_input = x
            # Get past KV for this layer if provided
            layer_past_key_value = past_key_values[i] if past_key_values is not None else None
            
            if self.use_checkpoint and self.training:
                # Pass arguments matching LLaDABlock.forward(self, x, attn_mask=None, use_kv_cache=False)
                # Note: use_kv_cache is set to False during training checkpointing
                # Checkpointing needs to handle past_kv and return present_kv correctly.
                # This might require a custom wrapper or disabling checkpointing for generation passes needing KV.
                # Assuming checkpoint handles it for now (might need adjustment):
                x, router_loss, present_key_value = checkpoint.checkpoint(block, block_input, attn_mask, layer_past_key_value, False, use_reentrant=False)
            else:
                # Standard forward pass
                x, router_loss, present_key_value = block(block_input, attn_mask=attn_mask, past_key_value=layer_past_key_value, use_kv_cache=False) # use_kv_cache=False typical for training/non-autoregressive passes
                 
            total_router_loss = total_router_loss + router_loss
            
            if return_kv_cache:
                present_key_values_list.append(present_key_value)
        
        # Apply final layer norm (common)
        x = self.ln_f(x)
        
        # Calculate logits (common)
        logits = self.lm_head(x) # Shape: [B, current_seq_len, V]
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            if use_bd3_training:
                # --- BD3-LM Loss Calculation ---
                # We need logits corresponding to the noisy input part
                noisy_logits = logits[:, seq_len:] # Shape: [B, seq_len, V]
                
                # Loss is calculated only on initially masked positions in the noisy part
                # using the original targets
                masked_indices_flat = masked_indices.view(-1) # From _bd3_noise_process call above
                
                # Use masked computation instead of conditional branching
                # Reshape logits and targets
                noisy_logits_flat = noisy_logits.reshape(-1, noisy_logits.size(-1)) # [B*seq_len, V]
                targets_flat = targets.reshape(-1) # [B*seq_len]
                
                # Calculate loss for all positions, then mask
                all_losses = F.cross_entropy(noisy_logits_flat, targets_flat, reduction='none')
                masked_losses = all_losses * masked_indices_flat.float()
                
                # Calculate mean only over masked positions
                num_masked = masked_indices_flat.float().sum()
                loss = torch.where(
                    num_masked > 0,
                    masked_losses.sum() / num_masked,
                    torch.tensor(0.0, device=device)
                )
                    
            elif apply_masking and masked_indices is not None:
                # --- Original LLaDA Loss Calculation ---
                # Calculate loss only on masked tokens (using p_mask weighting)
                masked_indices_flat = masked_indices.view(-1)
                # Use masked computation instead of conditional branching
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)
                p_mask_flat = p_mask.view(-1)
                
                # Calculate loss for all positions
                all_losses = F.cross_entropy(logits_flat, targets_flat, reduction='none')
                # Apply p_mask weighting and masked selection with clipping to prevent division by very large values
                # Clamp p_mask to prevent extreme weighting that causes mode collapse
                weighted_losses = (all_losses / torch.clamp(p_mask_flat, min=0.1, max=1.0)) * masked_indices_flat.float()
                
                # Calculate mean over all positions (normalized by batch_size * seq_len)
                loss = weighted_losses.sum() / (batch_size * seq_len)
                
                # Add entropy regularization to prevent mode collapse
                # This encourages the model to produce diverse predictions
                if self.training:  # Only apply during training
                    # Calculate entropy on masked positions only
                    masked_logits = logits_flat[masked_indices_flat]
                    if masked_logits.numel() > 0:
                        # Apply softmax to get probabilities
                        probs = F.softmax(masked_logits, dim=-1)
                        # Calculate entropy: -sum(p * log(p))
                        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                        # Average entropy across masked positions
                        avg_entropy = entropy.mean()
                        # Subtract from loss (higher entropy = lower loss)
                        # Use a small coefficient to not overwhelm the main loss
                        entropy_coef = 0.01
                        loss = loss - entropy_coef * avg_entropy
            # Else: No targets or no masking, loss remains None or 0.0 if initialized
            elif loss is None: # Ensure loss is tensor if targets provided but no masking
                loss = torch.tensor(0.0, device=device)
                 
        if return_kv_cache:
            return logits, loss, total_router_loss, present_key_values_list
        else:
            return logits, loss, total_router_loss
    
    # Renamed original generate method
    @torch.no_grad()
    def generate_original_llada(self, prompt, steps=32, gen_length=32, block_length=32, temperature=None, tokenizer=None, remasking='low_confidence'):
        """
        Generate text using the original LLaDA iterative demasking process.
        (Original docstring remains here)
        """
        # ... (Keep original implementation of generate here) ...
        # ... (Ensure self.forward calls inside use apply_masking=False) ...
        # Warning: Using original LLaDA generation logic.
        
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
                    # Use apply_masking=False for generation steps
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
                                masked_indices_local = torch.nonzero(seq_masked).squeeze(-1) # Indices within the gen region
                                unmask_positions_local = masked_indices_local[top_indices]
                                
                                # Sample from the model distribution
                                selected_probs = probs[b][unmask_positions_local]
                                sampled_tokens = torch.multinomial(selected_probs, 1).squeeze(-1)
                                
                                # Update the sequence (adjust indices for full sequence)
                                position_offset = prompt_length
                                generated[b, unmask_positions_local + position_offset] = sampled_tokens
                        
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
                                masked_indices_local = torch.nonzero(seq_masked).squeeze(-1) # Indices within the gen region
                                
                                # Randomly select positions to unmask
                                perm = torch.randperm(masked_indices_local.size(0), device=device)
                                unmask_positions_local = masked_indices_local[perm[:num_to_unmask]]
                                
                                # Sample from the model distribution
                                selected_probs = probs[b][unmask_positions_local]
                                sampled_tokens = torch.multinomial(selected_probs, 1).squeeze(-1)
                                
                                # Update the sequence (adjust indices for full sequence)
                                position_offset = prompt_length
                                generated[b, unmask_positions_local + position_offset] = sampled_tokens
            
            # Handle any remaining masked tokens
            remaining_masked = (generated == safe_mask_token_id)
            if remaining_masked.any():
                # Final forward pass to handle any remaining tokens
                final_logits, _, _ = self.forward(generated, apply_masking=False)
                final_probs = F.softmax(final_logits, dim=-1)
                
                # For each masked position, sample a replacement
                masked_indices_global = torch.nonzero(remaining_masked)
                for b, pos in masked_indices_global:
                    sampled_token = torch.multinomial(final_probs[b, pos], 1).item()
                    generated[b, pos] = sampled_token
            
            # Reset KV cache to free memory
            self.reset_kv_cache()
            
            return generated, None # Return None for aux data
            
        except Exception as e:
            # Error during generation
            # Reset KV cache in case of error
            self.reset_kv_cache()
            # Return original prompt or partial generation? Returning prompt for safety.
            return prompt, e 
    # Placeholder for the diffusion sampler for a single block
    def _sample_block_diffusion(self, model_fn, conditioning_kv=None, block_shape=None, device=None, steps=10, prev_tokens=None, repetition_penalty=1.2):
        """
        Samples a single block using a discrete diffusion process.
        Implements a simplified iterative denoising approach for LLaDA.

        Args:
            model_fn: A function that takes noisy block input (and optionally KV cache)
                      and returns logits for the clean block.
            conditioning_kv: Tuple (K, V) from previous blocks.
            block_shape: Tuple (batch_size, block_length).
            device: Torch device.
            steps: Number of diffusion steps for sampling this block.
            prev_tokens: Previously generated tokens for repetition penalty.
            repetition_penalty: Penalty factor for repeated tokens.

        Returns:
            sampled_block: Tensor of shape block_shape with sampled token IDs.
        """
        if block_shape is None or device is None:
            raise ValueError("block_shape and device must be provided.")
        
        batch_size, block_length = block_shape
        
        # Start with completely masked tokens (using mask token)
        # Ensure mask_token_id is within vocab range
        mask_token_id = min(self.config.mask_token_id, self.config.vocab_size - 1)
        current_tokens = torch.full(block_shape, mask_token_id, device=device, dtype=torch.long)
        
        # Iterative denoising: gradually unmask tokens
        for step in range(steps):
            # Calculate fraction of tokens to unmask in this step
            unmask_fraction = 1.0 - (steps - step - 1) / steps
            
            # Get model predictions for current state
            if conditioning_kv is not None:
                logits = model_fn(current_tokens, conditioning_kv)
            else:
                logits = model_fn(current_tokens)
            
            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Find currently masked positions
            masked_positions = (current_tokens == mask_token_id)
            
            if not masked_positions.any():
                break  # All tokens unmasked
            
            # For each sequence in batch
            for b in range(batch_size):
                seq_masked = masked_positions[b]
                if not seq_masked.any():
                    continue
                
                # Calculate how many tokens to unmask in this step
                total_masked = seq_masked.sum().item()
                target_unmasked_total = int(unmask_fraction * block_length)
                current_unmasked = block_length - total_masked
                tokens_to_unmask = max(0, min(total_masked, target_unmasked_total - current_unmasked))
                
                if tokens_to_unmask == 0:
                    continue
                
                # Get confidence scores for masked positions
                seq_probs = probs[b][seq_masked]
                confidence_scores = seq_probs.max(dim=-1).values
                
                # Select highest confidence positions to unmask
                _, top_indices = confidence_scores.topk(tokens_to_unmask)
                
                # Get the actual positions in the sequence
                masked_indices = torch.nonzero(seq_masked).squeeze(-1)
                positions_to_unmask = masked_indices[top_indices]
                
                # Sample tokens for these positions
                selected_probs = probs[b][positions_to_unmask]
                
                # Apply repetition penalty if previous tokens provided
                if prev_tokens is not None and repetition_penalty != 1.0:
                    # Get unique tokens from previous context
                    prev_unique = torch.unique(prev_tokens[b])
                    # Apply penalty to probabilities of repeated tokens
                    selected_probs[:, prev_unique] = selected_probs[:, prev_unique] / repetition_penalty
                    # Renormalize
                    selected_probs = selected_probs / selected_probs.sum(dim=-1, keepdim=True)
                
                sampled_tokens = torch.multinomial(selected_probs, 1).squeeze(-1)
                
                # Update the sequence
                current_tokens[b, positions_to_unmask] = sampled_tokens
        
        # Final pass: unmask any remaining masked tokens
        masked_positions = (current_tokens == mask_token_id)
        if masked_positions.any():
            if conditioning_kv is not None:
                logits = model_fn(current_tokens, conditioning_kv)
            else:
                logits = model_fn(current_tokens)
            
            probs = torch.softmax(logits, dim=-1)
            # Apply repetition penalty for final sampling
            if prev_tokens is not None and repetition_penalty != 1.0:
                for b in range(batch_size):
                    prev_unique = torch.unique(prev_tokens[b])
                    probs[b, :, prev_unique] = probs[b, :, prev_unique] / repetition_penalty
                    probs[b] = probs[b] / probs[b].sum(dim=-1, keepdim=True)
            
            # Sample from distribution for remaining masked positions
            # Handle the case where probs might have an extra dimension from the model
            if probs.dim() == 3:  # [batch_size, block_length, vocab_size]
                # Process each position individually
                for b in range(batch_size):
                    for pos in range(block_length):
                        if masked_positions[b, pos]:
                            # Apply repetition penalty if needed
                            pos_probs = probs[b, pos]
                            if prev_tokens is not None and repetition_penalty != 1.0:
                                prev_unique = torch.unique(prev_tokens[b])
                                pos_probs[prev_unique] = pos_probs[prev_unique] / repetition_penalty
                                pos_probs = pos_probs / pos_probs.sum()
                            # Sample token
                            sampled_token = torch.multinomial(pos_probs, 1).item()
                            current_tokens[b, pos] = sampled_token
            else:
                # Original logic for 2D probs
                final_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).squeeze(-1)
                final_tokens = final_tokens.view(batch_size, block_length)
                current_tokens = torch.where(masked_positions, final_tokens, current_tokens)
        
        return current_tokens
    
    @torch.no_grad()
    def generate(self, prompt, gen_length=128, temperature=None, top_k=None):
        """
        Generate text using the BD3-LM block-by-block diffusion sampling process.

        Args:
            prompt: Input token IDs [batch_size, prompt_length]
            gen_length: Total number of tokens to generate after the prompt.
            temperature: Temperature for sampling within the diffusion process (if applicable).
            top_k: Top-k sampling within the diffusion process (if applicable).

        Returns:
            generated: Generated token IDs [batch_size, prompt_length + gen_length]
        """
        self.eval() # Set model to evaluation mode
        device = prompt.device
        batch_size, prompt_length = prompt.shape
        
        # BD3-LM uses its specific block length
        block_length = self.bd3_block_length 
        num_gen_blocks = (gen_length + block_length - 1) // block_length
        total_gen_len_aligned = num_gen_blocks * block_length # Ensure generated length is multiple of block_length

        # BD3-LM Generation info
        
        # Add debug info
        safe_mask_token_id = min(self.config.mask_token_id, self.config.vocab_size - 1)
        # Using safe mask_token_id

        # Initialize the full sequence tensor
        full_seq = torch.zeros((batch_size, prompt_length + total_gen_len_aligned), dtype=torch.long, device=device)
        full_seq[:, :prompt_length] = prompt

        # --- KV Caching Setup ---
        # Store accumulated KV cache for each layer
        # Structure: List of (K, V) tuples, one per layer
        # K/V shape: [batch_size, n_head, accumulated_seq_len, head_size]
        accumulated_kv_cache = None 

        # --- Block-by-Block Generation ---
        for b in range(num_gen_blocks):
            # Generating block...
            start_idx = prompt_length + b * block_length
            end_idx = start_idx + block_length
            
            # --- Model function for diffusion sampling ---
            def model_fn(noisy_block_input, past_kv=None):
                # For BD3 generation, we only process the current block
                # The past context is maintained through KV cache
                
                # Run forward pass on just the noisy block
                with torch.no_grad():
                    # Enable KV cache for generation
                    self.enable_kv_cache()
                    
                    # Forward pass with past KV cache
                    logits, _, _, present_kv = self.forward(
                        input_ids=noisy_block_input,
                        targets=None,
                        apply_masking=False,  # Don't apply training masking
                        past_key_values=past_kv,
                        return_kv_cache=True
                    )
                    
                    # Disable KV cache after use
                    self.disable_kv_cache()
                
                # Return logits for the current block
                return logits

            # --- Sample the current block ---
            # Get previously generated tokens for repetition penalty
            prev_tokens = None
            if start_idx > 0:
                # Include prompt and all previously generated tokens
                prev_tokens = full_seq[:, :start_idx]
            
            sampled_block = self._sample_block_diffusion(
                model_fn=model_fn,
                conditioning_kv=accumulated_kv_cache,
                block_shape=(batch_size, block_length),
                device=device,
                steps=10, # Example diffusion steps per block
                prev_tokens=prev_tokens,
                repetition_penalty=1.2  # Apply repetition penalty
            )

            # Place the sampled block into the full sequence
            full_seq[:, start_idx:end_idx] = sampled_block

            # --- Update KV Cache ---
            # Run a forward pass on the clean sampled block to get its KV cache
            with torch.no_grad():
                # Process the prompt for the first block to initialize KV cache
                if b == 0 and prompt_length > 0:
                    # First, process the prompt to get initial KV cache
                    self.enable_kv_cache()
                    _, _, _, prompt_kv = self.forward(
                        input_ids=prompt,
                        targets=None,
                        apply_masking=False,
                        return_kv_cache=True
                    )
                    accumulated_kv_cache = prompt_kv
                    self.disable_kv_cache()
                
                # Now process the sampled block and update KV cache
                self.enable_kv_cache()
                _, _, _, block_kv = self.forward(
                    input_ids=sampled_block,
                    targets=None,
                    apply_masking=False,
                    past_key_values=accumulated_kv_cache,
                    return_kv_cache=True
                )
                # Update accumulated cache with new block's KV
                accumulated_kv_cache = block_kv
                self.disable_kv_cache()


        # Trim generated sequence to the requested gen_length
        final_generated_sequence = full_seq[:, :prompt_length + gen_length]
        
        return final_generated_sequence, None # Return None for loss/aux data
    def enable_kv_cache(self):
        """Enable KV caching for efficient generation"""
        for block in self.blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, 'kv_cache_enabled'):
                block.attn.kv_cache_enabled = True
                block.attn._cached_k = None
                block.attn._cached_v = None
    
    def disable_kv_cache(self):
        """Disable KV caching to free memory"""
        for block in self.blocks:
             if hasattr(block, 'attn') and hasattr(block.attn, 'kv_cache_enabled'):
                block.attn.kv_cache_enabled = False
                block.attn._cached_k = None
                block.attn._cached_v = None
    
    def reset_kv_cache(self):
        """Clear KV cache to free memory"""
        for block in self.blocks:
             if hasattr(block, 'attn') and hasattr(block.attn, 'kv_cache_enabled'):
                block.attn._cached_k = None
                block.attn._cached_v = None
        
        # Force CUDA memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def set_gradient_checkpointing(self, value):
        """Enable or disable gradient checkpointing"""
        self.use_checkpoint = value
    
    def estimate_mfu(self, batch_size: int, dt: float) -> float:
        """Estime l'utilisation des FLOPS du modèle (MFU) en pourcentage."""
        # Utiliser la fonction importée de train_utils.py
        # Passer le modèle, la taille du batch, la longueur de séquence, le temps d'exécution
        # et le type de données (fp16 pour les calculs en fp16)
        # Note: For BD3 training, seq_length might effectively be 2*seq_len? Needs verification.
        # Using config.block_size as a proxy for typical sequence length during training.
        return utils_estimate_mfu(
            model=self,
            batch_size=batch_size,
            seq_length=self.config.block_size, 
            dt=dt,
            dtype=torch.float16  # Utiliser fp16 comme demandé
        )
        
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, optimizer_type=None):
        """
        Configure optimizer with weight decay, specialized for LLaDA.
        
        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            betas: Adam beta parameters
            device_type: Device type ('cuda' or 'cpu')
            optimizer_type: Type of optimizer to use (default: 'apollo-mini')
            
        Returns:
            Configured optimizer
        """
        from ..optimizers import configure_optimizer_for_llada
        
        # Utiliser l'optimiseur spécifié ou le défaut pour LLaDA
        if optimizer_type is None:
            optimizer_type = 'apollo-mini'
            
        return configure_optimizer_for_llada(
            model=self,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
            device_type=device_type,
            optimizer_type=optimizer_type
        )