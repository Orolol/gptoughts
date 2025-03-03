"""
Adapter module for using LLaDA model with train_moe.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import os

from llada import LLaDAConfig, LLaDAModel

def init_llada_for_train_moe(model_args, device='cuda'):
    """
    Initialize a LLaDA model with configurations compatible with train_moe.py
    
    Args:
        model_args: A dictionary containing model configuration
        device: The device to place the model on
    
    Returns:
        Initialized LLaDA model
    """
    # Create LLaDA config from model_args
    config = LLaDAConfig(
        block_size=model_args.get('block_size', 1024),
        vocab_size=model_args.get('vocab_size', 50304),
        n_layer=model_args.get('n_layer', 12),
        n_head=model_args.get('n_head', 12),
        n_embd=model_args.get('n_embd', 768),
        dropout=model_args.get('dropout', 0.0),
        bias=model_args.get('bias', True),
        mask_token_id=model_args.get('mask_token_id', 126336),
        num_experts=model_args.get('num_experts', 8),
        k=model_args.get('k', 2)
    )
    
    # Initialize and return the model
    model = LLaDAModel(config)
    model.to(device)
    return model

def load_llada_from_checkpoint(checkpoint_path, device='cuda'):
    """
    Load a LLaDA model from a checkpoint in a format compatible with train_moe.py
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: The device to place the model on
    
    Returns:
        The loaded model and its configuration
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # If config not in checkpoint, use default config
        print("Warning: No config found in checkpoint, using default config")
        config = LLaDAConfig()
    
    # Create model with the extracted config
    model = LLaDAModel(config)
    
    # Load weights from checkpoint
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        # Try loading the checkpoint directly if it only contains model weights
        model.load_state_dict(checkpoint)
    
    model.to(device)
    return model, config

class LLaDAForTrainMoE(nn.Module):
    """
    A wrapper class for LLaDA model to make it compatible with train_moe.py
    """
    def __init__(self, llada_model):
        super().__init__()
        self.model = llada_model
        self.config = llada_model.config
        
    def forward(self, x, targets=None, apply_masking=True):
        """Forward method compatible with train_moe.py expectations"""
        
        with autocast(enabled=True, device_type='cuda'):
            # Use the LLaDA model's forward pass
            logits, loss, router_loss = self.model(x, targets=targets, apply_masking=apply_masking)
            return logits, loss, router_loss
    
    @torch.no_grad()
    def generate(self, x, max_new_tokens, temperature=None, top_k=None):
        """
        Generate method with interface compatible with train_moe.py's expectations
        """
        # Adapt to LLaDA generation parameters
        temperature = temperature if temperature is not None else self.config.temperature
        
        # Run LLaDA generation with transformed parameters
        generated = self.model.generate(
            prompt=x,
            steps=max_new_tokens,
            gen_length=max_new_tokens,
            block_length=min(128, max_new_tokens),  # Reasonable block size for generation
            temperature=temperature,
            remasking=self.config.remasking
        )
        
        return generated
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Pass through to LLaDA's optimizer configuration"""
        return self.model.configure_optimizers(weight_decay, learning_rate, betas, device_type)
    
    def set_gradient_checkpointing(self, value):
        """Pass through to LLaDA's gradient checkpointing configuration"""
        self.model.set_gradient_checkpointing(value) 