#!/usr/bin/env python3
"""
Test script to verify LLaDA BD3 configuration loads correctly
"""

import torch
import sys
from models.config import LLaDAConfig
from models.llada.model import LLaDAModel

def test_config():
    """Test that BD3 parameters are properly passed to the model"""
    
    # Create config with BD3 parameters
    config = LLaDAConfig(
        block_size=2048,
        vocab_size=128256,
        n_layer=16,
        n_head=16,
        n_embd=1024,
        dropout=0.0,
        bias=False,
        ratio_kv=8,
        use_checkpoint=False,
        bd3_block_length=128,
        bd3_beta=0.3,
        bd3_omega=0.8,
        disable_entropy_regularization=True
    )
    
    print("✓ Config created with BD3 parameters")
    
    # Create model
    model = LLaDAModel(config)
    print(f"✓ Model created successfully")
    
    # Check BD3 parameters
    assert model.bd3_block_length == 128, f"bd3_block_length mismatch: {model.bd3_block_length}"
    print(f"✓ bd3_block_length: {model.bd3_block_length}")
    
    assert hasattr(config, 'bd3_beta'), "bd3_beta not in config"
    assert config.bd3_beta == 0.3, f"bd3_beta mismatch: {config.bd3_beta}"
    print(f"✓ bd3_beta: {config.bd3_beta}")
    
    assert hasattr(config, 'bd3_omega'), "bd3_omega not in config"
    assert config.bd3_omega == 0.8, f"bd3_omega mismatch: {config.bd3_omega}"
    print(f"✓ bd3_omega: {config.bd3_omega}")
    
    assert hasattr(config, 'disable_entropy_regularization'), "disable_entropy_regularization not in config"
    assert config.disable_entropy_regularization == True, "entropy regularization not disabled"
    print(f"✓ disable_entropy_regularization: {config.disable_entropy_regularization}")
    
    # Test forward pass with BD3 training
    batch_size = 2
    seq_len = 512
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = input_ids.clone()
    
    print("\nTesting forward pass with BD3 training...")
    with torch.no_grad():
        logits, loss, router_loss = model(
            input_ids, 
            targets=targets, 
            use_bd3_training=True
        )
    
    assert logits is not None, "Logits is None"
    assert loss is not None, "Loss is None"
    assert not torch.isnan(loss), f"Loss is NaN: {loss}"
    assert not torch.isinf(loss), f"Loss is Inf: {loss}"
    
    print(f"✓ Forward pass successful")
    print(f"  - Loss value: {loss.item():.4f}")
    print(f"  - Logits shape: {logits.shape}")
    
    print("\n✅ All tests passed! BD3 configuration is working correctly.")

if __name__ == "__main__":
    test_config()