"""
Multi-Token Prediction (MTP) module for DeepSeek model.
This module implements the MTP capability described in the DeepSeek v3 architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any, Union

from models.deepseek.deepseek import (
    ModelArgs, Block, RMSNorm, ParallelEmbedding, 
    precompute_freqs_cis, apply_rotary_emb, Linear
)
from models.deepseek.deepseek_trainable import (
    TrainableBlock, TrainableMLA
)


class MTPTransformerBlock(TrainableBlock):
    """
    Transformer block tailored for Multi-Token Prediction (MTP).
    This is similar to the standard transformer block but with specific adaptations for MTP.
    """
    def __init__(self, layer_id: int, args: ModelArgs, mtp_idx: int = 0):
        """
        Initialize MTP transformer block.
        
        Args:
            layer_id: Layer index within the MTP module
            args: Model arguments containing dimensions and configuration
            mtp_idx: Index of this MTP module (0 for first future token, 1 for second, etc.)
        """
        super().__init__(layer_id, args)
        self.mtp_idx = mtp_idx


class MTPModule(nn.Module):
    """
    Multi-Token Prediction (MTP) module that predicts tokens beyond the next token.
    Each MTP module consists of one or more transformer layers designed to predict
    a specific future token (k+1, k+2, etc. where k is the MTP index).
    """
    def __init__(
        self, 
        args: ModelArgs, 
        mtp_idx: int, 
        num_layers: int = 1, 
        shared_embedding: Optional[nn.Module] = None,
        shared_norm: Optional[nn.Module] = None,
        shared_head: Optional[nn.Module] = None
    ):
        """
        Initialize an MTP module.
        
        Args:
            args: Model arguments containing dimensions and configuration
            mtp_idx: Index of this MTP module (0 for first future token, 1 for second, etc.)
            num_layers: Number of transformer layers in this MTP module
            shared_embedding: Embedding layer shared with the main model
            shared_norm: Final normalization layer shared with the main model
            shared_head: Output projection layer shared with the main model
        """
        super().__init__()
        self.args = args
        self.mtp_idx = mtp_idx
        
        # Create transformer blocks for this MTP module
        self.layers = nn.ModuleList([
            MTPTransformerBlock(layer_id=i, args=args, mtp_idx=mtp_idx)
            for i in range(num_layers)
        ])
        
        # Share embedding, normalization and output projection with the main model
        self.embed = shared_embedding
        self.norm = shared_norm if shared_norm is not None else RMSNorm(args.dim)
        self.head = shared_head
        
        # Register positional embeddings buffer
        self.register_buffer(
            "freqs_cis", 
            precompute_freqs_cis(args), 
            persistent=False
        )
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        input_ids: torch.Tensor,
        start_pos: int = 0,
        targets: Optional[torch.Tensor] = None,
        prev_mtp_output: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the MTP module.
        
        Args:
            hidden_states: Hidden states from the main model
            input_ids: Input token IDs
            start_pos: Starting position for positional embeddings
            targets: Target token IDs for computing loss
            prev_mtp_output: Output from the previous MTP module (if any)
            
        Returns:
            Tuple containing:
                - Output logits for token prediction
                - Loss value if targets are provided, otherwise None
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Incorporate the previous MTP output if available
        # This preserves the causal chain of MTP predictions
        if prev_mtp_output is not None:
            # Combine with previous MTP output
            hidden_states = hidden_states + prev_mtp_output
            
        # Get positional embeddings
        freqs_cis = self.freqs_cis[start_pos:start_pos+seq_len]
        
        # Create causal mask
        mask = None
        if seq_len > 1:
            mask = torch.full(
                (seq_len, seq_len), 
                float("-inf"), 
                device=hidden_states.device
            ).triu_(1)
            
        # Pass through transformer blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states, start_pos, freqs_cis, mask)
            
        # Apply final normalization
        mtp_output = self.norm(hidden_states)
        
        # Project to vocabulary space for token prediction
        if self.head is not None:
            logits = self.head(mtp_output)
        else:
            # Use a dummy projection if head is not provided (mainly for testing)
            logits = nn.Linear(
                self.args.dim, 
                self.args.vocab_size, 
                bias=False,
                device=hidden_states.device
            )(mtp_output)
            
        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            # For MTP-k, we need to shift targets k+1 positions
            shift_amount = self.mtp_idx + 1
            
            # Shift logits and targets for this MTP module
            # The idea is to predict the token that is (shift_amount) steps ahead
            if seq_len > shift_amount:
                shift_logits = logits[:, :-shift_amount, :]
                shift_targets = targets[:, shift_amount:]
                
                # Compute cross entropy loss
                loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_targets.reshape(-1),
                    ignore_index=-1
                )
                
                # Check for NaN loss
                if torch.isnan(loss).any():
                    loss = torch.where(torch.isnan(loss), torch.full_like(loss, 100.0), loss)
                
                # Cap loss to avoid gradient explosions
                loss = torch.clamp(loss, max=100.0)
                
        return logits, loss, mtp_output


class MTPHead(nn.Module):
    """
    Complete Multi-Token Prediction module.
    This contains multiple MTP modules, each predicting a different future token.
    """
    def __init__(
        self, 
        args: ModelArgs, 
        num_mtp_modules: int = 1,
        layers_per_mtp: int = 1,
        shared_embedding: Optional[nn.Module] = None,
        shared_norm: Optional[nn.Module] = None,
        shared_head: Optional[nn.Module] = None
    ):
        """
        Initialize the complete MTP head.
        
        Args:
            args: Model arguments containing dimensions and configuration
            num_mtp_modules: Number of MTP modules to include
            layers_per_mtp: Number of transformer layers in each MTP module
            shared_embedding: Embedding layer shared with the main model
            shared_norm: Final normalization layer shared with the main model
            shared_head: Output projection layer shared with the main model
        """
        super().__init__()
        self.args = args
        self.num_mtp_modules = num_mtp_modules
        
        # Create individual MTP modules
        self.mtp_modules = nn.ModuleList([
            MTPModule(
                args=args,
                mtp_idx=i,
                num_layers=layers_per_mtp,
                shared_embedding=shared_embedding,
                shared_norm=shared_norm,
                shared_head=shared_head
            )
            for i in range(num_mtp_modules)
        ])
        
        # Regularization factor for MTP loss
        self.loss_factor = 0.1
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        input_ids: torch.Tensor,
        start_pos: int = 0,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for the complete MTP head.
        
        Args:
            hidden_states: Hidden states from the main model
            input_ids: Input token IDs
            start_pos: Starting position for positional embeddings
            targets: Target token IDs for computing loss
            
        Returns:
            Tuple containing:
                - List of output logits from each MTP module
                - Combined MTP loss if targets are provided, otherwise None
        """
        outputs = []
        losses = []
        prev_output = None
        
        # Process through each MTP module in sequence
        for i, mtp_module in enumerate(self.mtp_modules):
            logits, loss, mtp_output = mtp_module(
                hidden_states=hidden_states,
                input_ids=input_ids,
                start_pos=start_pos,
                targets=targets,
                prev_mtp_output=prev_output
            )
            
            outputs.append(logits)
            if loss is not None:
                losses.append(loss)
                
            # Pass the output to the next MTP module in the chain
            prev_output = mtp_output
            
        # Combine losses if available
        combined_loss = None
        if losses:
            combined_loss = sum(losses) / len(losses) * self.loss_factor
            
        return outputs, combined_loss
        
    def generate_speculative(
        self, 
        hidden_states: torch.Tensor, 
        input_ids: torch.Tensor,
        start_pos: int = 0,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> List[int]:
        """
        Generate tokens speculatively using MTP modules.
        
        Args:
            hidden_states: Hidden states from the main model
            input_ids: Input token IDs
            start_pos: Starting position for positional embeddings
            temperature: Temperature for sampling
            top_k: Number of top tokens to consider for sampling
            
        Returns:
            List of predicted token IDs
        """
        with torch.no_grad():
            # Get predictions from each MTP module
            predicted_tokens = []
            prev_output = None
            
            for i, mtp_module in enumerate(self.mtp_modules):
                logits, _, mtp_output = mtp_module(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    start_pos=start_pos,
                    prev_mtp_output=prev_output
                )
                
                # Sample from the logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k sampling if specified
                if top_k is not None and top_k > 0:
                    indices_to_remove = torch.topk(next_token_logits, top_k)[0][:, -1].unsqueeze(-1)
                    next_token_logits[next_token_logits < indices_to_remove] = float('-inf')
                    
                # Sample token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                predicted_tokens.append(next_token.item())
                prev_output = mtp_output
                
            return predicted_tokens