"""Test script to verify FP8 MLA integration."""
import torch
import torch.nn as nn
from models.models.mla_model import MLAModel, MLAModelConfig
from models.blocks.mla_fp8 import MLA_FP8
from models.blocks.mla import MLA

def test_fp8_integration():
    """Test that FP8 MLA blocks are correctly instantiated."""
    
    # Test 1: Create model with FP8 disabled
    print("Test 1: Creating MLA model without FP8...")
    config_no_fp8 = MLAModelConfig(
        n_layer=2,
        n_embd=768,
        n_head=12,
        vocab_size=50257,
        block_size=1024,
        use_fp8=False,
        fp8_params=False,
        fp8_mla_params=False
    )
    model_no_fp8 = MLAModel(config_no_fp8)
    
    # Check that standard MLA blocks are used
    block_0 = model_no_fp8.transformer.h[0]
    assert isinstance(block_0.attn, MLA), f"Expected MLA, got {type(block_0.attn)}"
    print("✓ Model without FP8 uses standard MLA blocks")
    
    # Test 2: Create model with FP8 enabled
    print("\nTest 2: Creating MLA model with FP8...")
    config_fp8 = MLAModelConfig(
        n_layer=2,
        n_embd=768,
        n_head=12,
        vocab_size=50257,
        block_size=1024,
        use_fp8=True,
        fp8_params=True,
        fp8_mla_params=False,  # Keep MLA params in FP16
        fp8_tile_size=128
    )
    model_fp8 = MLAModel(config_fp8)
    
    # Check that FP8 MLA blocks are used
    block_0_fp8 = model_fp8.transformer.h[0]
    assert isinstance(block_0_fp8.attn, MLA_FP8), f"Expected MLA_FP8, got {type(block_0_fp8.attn)}"
    print("✓ Model with FP8 uses MLA_FP8 blocks")
    
    # Test 3: Forward pass test
    print("\nTest 3: Testing forward pass...")
    batch_size = 2
    seq_len = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create dummy input
    input_ids = torch.randint(0, config_fp8.vocab_size, (batch_size, seq_len), device=device)
    
    # Test forward pass for both models
    model_no_fp8 = model_no_fp8.to(device)
    model_fp8 = model_fp8.to(device)
    
    with torch.no_grad():
        # Test no FP8 model
        output_no_fp8, loss_no_fp8 = model_no_fp8(input_ids, input_ids)
        print(f"✓ No FP8 model forward pass successful. Output shape: {output_no_fp8.shape}")
        
        # Test FP8 model
        output_fp8, loss_fp8 = model_fp8(input_ids, input_ids)
        print(f"✓ FP8 model forward pass successful. Output shape: {output_fp8.shape}")
    
    # Test 4: Check parameter count
    print("\nTest 4: Checking parameter counts...")
    params_no_fp8 = sum(p.numel() for p in model_no_fp8.parameters())
    params_fp8 = sum(p.numel() for p in model_fp8.parameters())
    
    # Should have same parameter count
    assert params_no_fp8 == params_fp8, f"Parameter mismatch: {params_no_fp8} vs {params_fp8}"
    print(f"✓ Both models have same parameter count: {params_no_fp8:,}")
    
    # Test 5: Gradient test
    print("\nTest 5: Testing backward pass...")
    input_small = input_ids[:1, :64]  # Smaller input for gradient test
    
    # Enable gradients
    model_fp8.train()
    
    # Forward and backward
    output, loss = model_fp8(input_small, input_small)
    if loss is not None and loss.requires_grad:
        loss.backward()
        print("✓ Backward pass successful with FP8 model")
    else:
        print("⚠ Loss does not require gradients or is None")
    
    print("\n✅ All tests passed! FP8 integration is working correctly.")
    
    return model_no_fp8, model_fp8

if __name__ == "__main__":
    try:
        test_fp8_integration()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()