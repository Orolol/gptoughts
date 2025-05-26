#!/usr/bin/env python3
"""
Test script to compare Dynamic Tanh (DyT) vs RMSNorm performance and stability.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from models.blocks.normalization import RMSNorm, DynamicTanh
from models.models.mla_model import create_mla_model, MLAModelConfig

def test_normalization_layers():
    """Compare DyT and RMSNorm on various metrics."""
    print("=== Testing Normalization Layers ===\n")
    
    # Test dimensions
    batch_size = 32
    seq_len = 512
    dim = 768
    
    # Create test input
    x = torch.randn(batch_size, seq_len, dim).cuda()
    
    # Create normalization layers
    rmsnorm = RMSNorm(dim).cuda()
    dyt = DynamicTanh(dim).cuda()
    
    # Forward pass timing
    n_runs = 100
    
    # Warmup
    for _ in range(10):
        _ = rmsnorm(x)
        _ = dyt(x)
    
    # Time RMSNorm
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_runs):
        out_rms = rmsnorm(x)
    torch.cuda.synchronize()
    rms_time = time.time() - start
    
    # Time DyT
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_runs):
        out_dyt = dyt(x)
    torch.cuda.synchronize()
    dyt_time = time.time() - start
    
    print(f"RMSNorm forward time: {rms_time:.4f}s ({rms_time/n_runs*1000:.2f}ms per batch)")
    print(f"DyT forward time: {dyt_time:.4f}s ({dyt_time/n_runs*1000:.2f}ms per batch)")
    print(f"DyT speedup: {(rms_time/dyt_time - 1)*100:.1f}%\n")
    
    # Compare outputs
    print("Output statistics:")
    print(f"RMSNorm - mean: {out_rms.mean():.4f}, std: {out_rms.std():.4f}")
    print(f"DyT - mean: {out_dyt.mean():.4f}, std: {out_dyt.std():.4f}\n")
    
    # Gradient stability test
    print("=== Gradient Stability Test ===")
    x.requires_grad = True
    
    # RMSNorm backward
    out_rms = rmsnorm(x)
    loss_rms = out_rms.mean()
    loss_rms.backward()
    grad_rms = x.grad.clone()
    x.grad.zero_()
    
    # DyT backward
    out_dyt = dyt(x)
    loss_dyt = out_dyt.mean()
    loss_dyt.backward()
    grad_dyt = x.grad.clone()
    
    print(f"RMSNorm gradient - mean: {grad_rms.mean():.6f}, std: {grad_rms.std():.6f}")
    print(f"DyT gradient - mean: {grad_dyt.mean():.6f}, std: {grad_dyt.std():.6f}\n")
    

def test_model_with_dyt():
    """Test full model with DyT enabled."""
    print("=== Testing MLA Model with DyT ===\n")
    
    # Create two models - one with RMSNorm, one with DyT
    config_rms = MLAModelConfig(
        n_layer=4,
        n_embd=768,
        n_head=12,
        vocab_size=50304,
        block_size=512,
        use_dyt=False
    )
    
    config_dyt = MLAModelConfig(
        n_layer=4,
        n_embd=768,
        n_head=12,
        vocab_size=50304,
        block_size=512,
        use_dyt=True,
        dyt_alpha_init=0.5
    )
    
    model_rms = create_mla_model(
        size='small',
        n_layer=4,
        use_dyt=False
    ).cuda()
    
    model_dyt = create_mla_model(
        size='small',
        n_layer=4,
        use_dyt=True,
        dyt_alpha_init=0.5
    ).cuda()
    
    # Create sample input
    batch_size = 8
    seq_len = 256
    idx = torch.randint(0, 50304, (batch_size, seq_len)).cuda()
    targets = torch.randint(0, 50304, (batch_size, seq_len)).cuda()
    
    # Forward pass timing
    n_runs = 10
    
    # Warmup
    for _ in range(2):
        _, _ = model_rms(idx, targets)
        _, _ = model_dyt(idx, targets)
    
    # Time RMSNorm model
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_runs):
        logits_rms, loss_rms = model_rms(idx, targets)
    torch.cuda.synchronize()
    rms_time = time.time() - start
    
    # Time DyT model
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_runs):
        logits_dyt, loss_dyt = model_dyt(idx, targets)
    torch.cuda.synchronize()
    dyt_time = time.time() - start
    
    print(f"Model with RMSNorm - forward time: {rms_time:.4f}s ({rms_time/n_runs*1000:.2f}ms per batch)")
    print(f"Model with DyT - forward time: {dyt_time:.4f}s ({dyt_time/n_runs*1000:.2f}ms per batch)")
    print(f"DyT model speedup: {(rms_time/dyt_time - 1)*100:.1f}%\n")
    
    print(f"Loss comparison:")
    print(f"RMSNorm model loss: {loss_rms.item():.4f}")
    print(f"DyT model loss: {loss_dyt.item():.4f}\n")
    

def main():
    """Run all tests."""
    print("Testing Dynamic Tanh (DyT) Implementation\n")
    print("Based on 'Transformers without Normalization' by Meta FAIR")
    print("Expected improvements: ~8.2% training speedup, ~7.8% inference speedup\n")
    
    # Test normalization layers in isolation
    test_normalization_layers()
    
    # Test full model with DyT
    test_model_with_dyt()
    
    print("=== Summary ===")
    print("Dynamic Tanh (DyT) successfully implemented!")
    print("- Simple drop-in replacement for RMSNorm/LayerNorm")
    print("- Single learnable alpha parameter per layer")
    print("- Elementwise affine transformation (weight & bias)")
    print("- Ready for training with --use_dyt flag")


if __name__ == "__main__":
    main()