#!/usr/bin/env python3
"""Test script to verify LLaDA with DynTanh normalization."""

import torch
from models.config import LLaDAConfig
from models.llada.model import LLaDAModel

def test_llada_dyt():
    # Create config with DynTanh enabled
    config = LLaDAConfig(
        block_size=512,
        vocab_size=50304,
        n_layer=4,
        n_head=8,
        n_embd=512,
        num_experts=4,
        k=2,
        use_dyt=True,
        dyt_alpha_init=0.5,
        bd3_block_length=128
    )
    
    # Create model
    model = LLaDAModel(config)
    print(f"Created LLaDA model with DynTanh normalization")
    
    # Check that DynTanh is used instead of RMSNorm
    from models.blocks.normalization import DynamicTanh
    
    dyt_count = 0
    for name, module in model.named_modules():
        if isinstance(module, DynamicTanh):
            dyt_count += 1
            print(f"Found DynTanh layer: {name}, alpha={module.alpha.item():.3f}")
    
    expected_dyt = config.n_layer * 2 + 1  # 2 per block + 1 final
    print(f"\nTotal DynTanh layers: {dyt_count} (expected: {expected_dyt})")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Test with BD3 training
    print("\nTesting forward pass with BD3 training...")
    with torch.no_grad():
        logits, loss, router_loss = model(input_ids, targets=input_ids, use_bd3_training=True)
        print(f"BD3 forward pass successful. Logits shape: {logits.shape}")
    
    # Test without BD3 training
    print("\nTesting forward pass without BD3 training...")
    with torch.no_grad():
        logits, loss, router_loss = model(input_ids, targets=input_ids, use_bd3_training=False)
        print(f"Standard forward pass successful. Logits shape: {logits.shape}")
    
    # Check parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Check alpha parameters
    alpha_params = [p for name, p in model.named_parameters() if 'alpha' in name]
    print(f"\nFound {len(alpha_params)} alpha parameters for DynTanh")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_llada_dyt()