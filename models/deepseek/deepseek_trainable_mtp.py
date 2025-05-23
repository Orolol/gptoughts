"""
DeepSeek trainable model with Multi-Token Prediction (MTP) support.
This implementation extends the trainable model to include MTP capabilities.
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal, Dict, Any, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from models.deepseek.deepseek import ModelArgs
from models.deepseek.deepseek_trainable import DeepSeekTrainable, TrainableTransformer
from models.deepseek.mtp import MTPHead


class MTPModelArgs(ModelArgs):
    """Extension of ModelArgs with MTP-specific parameters."""
    
    def __init__(
        self,
        *args,
        num_mtp_modules: int = 1,
        layers_per_mtp: int = 1,
        mtp_loss_factor: float = 0.1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_mtp_modules = num_mtp_modules
        self.layers_per_mtp = layers_per_mtp
        self.mtp_loss_factor = mtp_loss_factor


class DeepSeekTrainableMTP(nn.Module):
    """
    Extended DeepSeek model with Multi-Token Prediction (MTP) support.
    
    This class adds MTP modules to the DeepSeek model, enabling prediction of 
    multiple tokens ahead and supporting speculative decoding during inference.
    """
    
    def __init__(self, args: MTPModelArgs):
        super().__init__()
        
        # Create the base DeepSeek model
        self.base_model = DeepSeekTrainable(args)
        
        # Store parameters
        self.args = args
        self.training_mode = False
        self.gradient_checkpointing = False
        
        # Create MTP head that encapsulates all MTP modules
        # Share the embedding, final norm and output projection with the base model
        self.mtp_head = MTPHead(
            args=args,
            num_mtp_modules=args.num_mtp_modules,
            layers_per_mtp=args.layers_per_mtp,
            shared_embedding=self.base_model.transformer.embed,
            shared_norm=self.base_model.transformer.norm,
            shared_head=self.base_model.transformer.head
        )
        
        # Set the MTP loss factor
        self.mtp_head.loss_factor = args.mtp_loss_factor
    
    def set_gradient_checkpointing(self, value: bool):
        """Enable or disable gradient checkpointing."""
        self.gradient_checkpointing = value
        self.base_model.set_gradient_checkpointing(value)
        return self
    
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_mtp: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass combining base model and MTP modules.
        
        Args:
            input_ids: Input token IDs
            targets: Target token IDs for computing loss
            use_mtp: Whether to use MTP for training/inference
            **kwargs: Additional arguments
            
        Returns:
            In training mode: tuple of (logits, loss) or (logits, (loss, mtp_loss))
            In inference mode: just logits
        """
        # Set training mode based on the current training state
        self.base_model.transformer.set_training_mode(self.training)
        
        # Extract start position if provided
        start_pos = kwargs.get('start_pos', 0)
        
        # Run the base transformer model
        base_outputs = self.base_model(input_ids, targets=targets, start_pos=start_pos)
        
        # Handle different return formats based on training/inference
        if self.training and targets is not None:
            base_logits, base_loss = base_outputs
            
            # Get the hidden states from the base model
            # We need to re-compute them since they're not returned in the output
            with torch.set_grad_enabled(True):
                # Get embeddings
                h = self.base_model.transformer.embed(input_ids)
                
                # Get positional embeddings
                freqs_cis = self.base_model.transformer.freqs_cis[start_pos:start_pos+input_ids.size(1)]
                
                # Create causal mask
                mask = None
                if input_ids.size(1) > 1:
                    mask = torch.full(
                        (input_ids.size(1), input_ids.size(1)), 
                        float("-inf"), 
                        device=input_ids.device
                    ).triu_(1)
                
                # Pass through transformer layers
                for layer in self.base_model.transformer.layers:
                    h = layer(h, start_pos, freqs_cis, mask)
                
                # Get the final hidden states
                hidden_states = h
            
            # If MTP is enabled and we're training
            if use_mtp:
                # Forward pass through MTP modules
                mtp_outputs, mtp_loss = self.mtp_head(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    start_pos=start_pos,
                    targets=targets
                )
                
                # Combine base loss and MTP loss
                if mtp_loss is not None:
                    total_loss = base_loss + mtp_loss
                    # Return logits and combined loss
                    return base_logits, (total_loss, mtp_loss)
            
            # Return base model outputs if MTP is not used
            return base_logits, base_loss
        
        else:
            # In inference mode, just return the logits
            return base_outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_speculative: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text with optional speculative decoding using MTP.
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            top_k: Number of top tokens to consider for sampling
            use_speculative: Whether to use speculative decoding with MTP
            **kwargs: Additional arguments
            
        Returns:
            Tensor of generated token IDs
        """
        # Put model in eval mode
        self.base_model.eval()
        self.eval()
        
        # Extract parameters
        start_pos = kwargs.get('start_pos', 0)
        batch_size = input_ids.size(0)
        
        # Use autoregressive generation by default
        if not use_speculative or self.args.num_mtp_modules < 1:
            return self._generate_autoregressive(
                input_ids, 
                max_new_tokens, 
                temperature, 
                top_k, 
                start_pos
            )
        
        # Speculative decoding using MTP
        for _ in range(0, max_new_tokens, self.args.num_mtp_modules + 1):
            # Get base model hidden states for current sequence
            with torch.no_grad():
                # Get embeddings
                h = self.base_model.transformer.embed(input_ids)
                
                # Get positional embeddings
                seq_len = input_ids.size(1)
                freqs_cis = self.base_model.transformer.freqs_cis[start_pos:start_pos+seq_len]
                
                # Create causal mask
                mask = None
                if seq_len > 1:
                    mask = torch.full(
                        (seq_len, seq_len), 
                        float("-inf"), 
                        device=input_ids.device
                    ).triu_(1)
                
                # Pass through transformer layers
                for layer in self.base_model.transformer.layers:
                    h = layer(h, start_pos, freqs_cis, mask)
                
                # Apply final normalization
                hidden_states = self.base_model.transformer.norm(h)
            
            # Generate the next token non-speculatively
            next_token_logits = self.base_model.transformer.head(hidden_states[:, -1:])
            
            # Sample from the logits
            next_token = self._sample_token(next_token_logits[:, -1], temperature, top_k)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=1)
            
            # Check if we've reached the token limit
            if input_ids.size(1) >= max_new_tokens:
                break
                
            # Generate speculative tokens using MTP modules
            speculative_tokens = self.mtp_head.generate_speculative(
                hidden_states=hidden_states,
                input_ids=input_ids,
                start_pos=start_pos + seq_len,
                temperature=temperature,
                top_k=top_k
            )
            
            # Verify the speculative tokens
            accepted_tokens = []
            for i, token in enumerate(speculative_tokens):
                # Add token to input_ids for verification
                test_input_ids = torch.cat([
                    input_ids, 
                    torch.tensor([token], device=input_ids.device).unsqueeze(0)
                ], dim=1)
                
                # Get the base model's prediction for this position
                with torch.no_grad():
                    base_logits = self.base_model(test_input_ids).squeeze(0)
                
                # Sample from base model logits
                base_token = self._sample_token(
                    base_logits[-1], 
                    temperature, 
                    top_k
                ).item()
                
                # If the base model agrees with the speculative token, accept it
                if base_token == token:
                    accepted_tokens.append(token)
                else:
                    # Stop at the first disagreement
                    accepted_tokens.append(base_token)
                    break
            
            # Add accepted tokens to input_ids
            if accepted_tokens:
                accepted_tensor = torch.tensor(
                    accepted_tokens, 
                    device=input_ids.device
                ).unsqueeze(0)
                input_ids = torch.cat([input_ids, accepted_tensor], dim=1)
            
            # Check if we've reached the token limit
            if input_ids.size(1) >= max_new_tokens:
                break
                
            # Update start_pos
            start_pos += input_ids.size(1) - seq_len
        
        # Ensure we don't exceed max_new_tokens
        if input_ids.size(1) > max_new_tokens:
            input_ids = input_ids[:, :max_new_tokens]
            
        return input_ids
    
    def _generate_autoregressive(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        start_pos: int
    ) -> torch.Tensor:
        """Standard autoregressive generation without speculation."""
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                logits = self.base_model(input_ids, start_pos=start_pos)
            
            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample from the logits
            next_token = self._sample_token(next_token_logits, temperature, top_k)
            
            # Add token to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=1)
            
            # Update start_pos if needed
            start_pos += 1
            
        return input_ids
    
    def _sample_token(
        self, 
        logits: torch.Tensor, 
        temperature: float, 
        top_k: Optional[int]
    ) -> torch.Tensor:
        """Helper method to sample a token from logits."""
        if temperature == 0:
            # Greedy decoding
            return torch.argmax(logits, dim=-1)
        
        # Apply temperature
        logits = logits / max(temperature, 1e-5)
        
        # Apply top-k sampling if specified
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
            
        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        return next_token
    
    def configure_optimizers(
        self, 
        weight_decay: float, 
        learning_rate: float, 
        betas: Tuple[float, float], 
        device_type: str, 
        **kwargs
    ):
        """Configure the optimizer for training."""
        # Collect all parameters 
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Create separate groups for parameters with and without weight decay
        no_decay = ['bias', 'norm', 'embedding']
        optimizer_grouped_params = [
            {
                'params': [p for n, p in param_dict.items() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in param_dict.items() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        # Get optimizer type if specified
        optimizer_type = kwargs.get('optimizer_type', 'adamw')
        
        # Determine if fused implementation is available
        fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type == 'cuda'
        
        # Create optimizer based on type
        if optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                optimizer_grouped_params,
                lr=learning_rate,
                betas=betas,
                fused=use_fused if use_fused else False,
            )
        elif optimizer_type.lower() == 'lion':
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
    
    def estimate_mfu(self, batch_size: int, seq_length: int, dt: float) -> float:
        """Estimate model flops utilization (MFU)."""
        # Include MTP modules in flops calculation
        # Constants and model parameters
        N = sum(p.numel() for p in self.parameters())
        L_base = self.args.n_layers
        L_mtp = self.args.num_mtp_modules * self.args.layers_per_mtp
        L_total = L_base + L_mtp
        H = self.args.dim
        
        # Flops estimate for base model + MTP modules
        flops_per_token = 6 * N + 12 * L_total * H * seq_length
        flops = flops_per_token * batch_size
        
        # Estimate achieved flops
        flops_achieved = flops / (dt * 1e12)  # Expressed in TFLOPs
        
        # Theoretical peak flops for A100 GPU (adjust based on hardware)
        theoretical_flops = 312  # A100 GPU theoretical TFLOPs for BF16
        
        # MFU calculation
        mfu = flops_achieved / theoretical_flops
        return mfu