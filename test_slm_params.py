#!/usr/bin/env python3
"""
SLM Parameter Count Test and Analysis Script

This script validates that the SLM-MoE-MLA model meets the target parameter count
and provides detailed analysis of the model architecture.
"""

import sys
import torch
import torch.nn as nn
from typing import Dict, Tuple
import argparse

# Add project root to path
sys.path.append('.')

from models.models.slm_moe_mla import create_slm_model, SLMConfig, SLMMLA

def count_parameters(model: nn.Module) -> Tuple[int, int, int]:
    """
    Count total, trainable, and non-trainable parameters.
    
    Returns:
        total_params, trainable_params, non_trainable_params
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    
    return total, trainable, non_trainable

def analyze_parameter_distribution(model: SLMMLA) -> Dict[str, int]:
    """Analyze parameter distribution across different components."""
    param_counts = {
        'embeddings': 0,
        'mla_attention': 0,
        'shared_mlp': 0,
        'expert_mlp': 0,
        'routers': 0,
        'normalization': 0,
        'output_head': 0,
        'other': 0
    }
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        
        if 'wte' in name or 'lm_head' in name:
            if 'wte' in name:
                param_counts['embeddings'] += param_count
            else:
                param_counts['output_head'] += param_count
        elif 'attn' in name:
            param_counts['mla_attention'] += param_count
        elif 'shared_up_proj' in name or 'shared_down_proj' in name or 'mix_weights' in name:
            param_counts['shared_mlp'] += param_count
        elif 'expert_up_proj' in name or 'expert_down_proj' in name:
            param_counts['expert_mlp'] += param_count
        elif 'router' in name:
            param_counts['routers'] += param_count
        elif 'ln_' in name or 'norm' in name:
            param_counts['normalization'] += param_count
        else:
            param_counts['other'] += param_count
    
    return param_counts

def calculate_moe_efficiency(model: SLMMLA) -> Dict[str, float]:
    """Calculate MoE efficiency metrics."""
    config = model.config
    
    # Count expert parameters
    expert_params = 0
    shared_params = 0
    
    param_dist = analyze_parameter_distribution(model)
    expert_params = param_dist['expert_mlp']
    total_params = sum(param_dist.values())
    
    # Calculate effective parameters during inference
    # Only a subset of experts are used per token
    effective_expert_params = expert_params / config.num_experts * config.experts_per_token
    shared_and_other_params = total_params - expert_params
    effective_total_params = shared_and_other_params + effective_expert_params
    
    efficiency_metrics = {
        'total_params_m': total_params / 1e6,
        'effective_params_m': effective_total_params / 1e6,
        'expert_params_m': expert_params / 1e6,
        'shared_params_m': shared_and_other_params / 1e6,
        'parameter_efficiency': effective_total_params / total_params,
        'memory_efficiency': 1.0 - (expert_params * (1.0 - config.experts_per_token / config.num_experts)) / total_params,
        'expert_utilization_ratio': config.experts_per_token / config.num_experts
    }
    
    return efficiency_metrics

def test_model_forward(model: SLMMLA, device: str = 'cpu') -> bool:
    """Test basic forward pass to ensure model works correctly."""
    try:
        model.to(device)
        model.eval()
        
        # Create dummy input
        batch_size, seq_len = 2, 128
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len)).to(device)
        targets = torch.randint(0, model.config.vocab_size, (batch_size, seq_len)).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, targets=targets)
            
            # Check outputs
            assert 'logits' in outputs, "Model should return logits"
            assert 'loss' in outputs, "Model should return loss when targets provided"
            assert 'router_loss' in outputs, "Model should return router_loss"
            
            logits = outputs['logits']
            loss = outputs['loss']
            router_loss = outputs['router_loss']
            
            # Check shapes
            assert logits.shape == (batch_size, seq_len, model.config.vocab_size), f"Unexpected logits shape: {logits.shape}"
            assert loss.dim() == 0, f"Loss should be scalar, got shape: {loss.shape}"
            assert router_loss.dim() == 0, f"Router loss should be scalar, got shape: {router_loss.shape}"
            
            # Check for NaN values
            assert not torch.isnan(logits).any(), "NaN detected in logits"
            assert not torch.isnan(loss).any(), "NaN detected in loss"
            assert not torch.isnan(router_loss).any(), "NaN detected in router_loss"
            
            print(f"✅ Forward pass test passed!")
            print(f"   Logits shape: {logits.shape}")
            print(f"   Loss: {loss.item():.4f}")
            print(f"   Router loss: {router_loss.item():.6f}")
            
            return True
            
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        return False

def run_parameter_tests(model_size: str = 'small', target_params_m: float = 200.0, 
                       tolerance: float = 0.1, device: str = 'cpu'):
    """Run comprehensive parameter count and model tests."""
    
    print("=" * 80)
    print("SLM-MoE-MLA Parameter Analysis and Testing")
    print("=" * 80)
    
    # Create model
    print(f"Creating SLM model (size: {model_size})...")
    try:
        model = create_slm_model(model_size)
        print("✅ Model creation successful")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # Count parameters
    total_params, trainable_params, non_trainable_params = count_parameters(model)
    
    print(f"\n📊 Parameter Count Summary:")
    print(f"   Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"   Non-trainable parameters: {non_trainable_params:,} ({non_trainable_params/1e6:.2f}M)")
    
    # Check if we meet target
    actual_params_m = total_params / 1e6
    diff_from_target = abs(actual_params_m - target_params_m)
    within_tolerance = diff_from_target <= target_params_m * tolerance
    
    print(f"\n🎯 Target Validation:")
    print(f"   Target: {target_params_m:.1f}M parameters")
    print(f"   Actual: {actual_params_m:.2f}M parameters") 
    print(f"   Difference: {diff_from_target:.2f}M ({diff_from_target/target_params_m*100:.1f}%)")
    print(f"   Within tolerance ({tolerance*100:.1f}%): {'✅ Yes' if within_tolerance else '❌ No'}")
    
    # Detailed parameter distribution
    param_dist = analyze_parameter_distribution(model)
    print(f"\n🔍 Parameter Distribution:")
    total_for_check = sum(param_dist.values())
    for component, count in param_dist.items():
        percentage = count / total_for_check * 100
        print(f"   {component.replace('_', ' ').title()}: {count/1e6:.2f}M ({percentage:.1f}%)")
    
    # MoE efficiency analysis
    efficiency = calculate_moe_efficiency(model)
    print(f"\n⚡ MoE Efficiency Analysis:")
    print(f"   Total parameters: {efficiency['total_params_m']:.2f}M")
    print(f"   Effective parameters: {efficiency['effective_params_m']:.2f}M")
    print(f"   Expert parameters: {efficiency['expert_params_m']:.2f}M")
    print(f"   Shared parameters: {efficiency['shared_params_m']:.2f}M")
    print(f"   Parameter efficiency: {efficiency['parameter_efficiency']*100:.1f}%")
    print(f"   Memory efficiency: {efficiency['memory_efficiency']*100:.1f}%")
    print(f"   Expert utilization: {efficiency['expert_utilization_ratio']*100:.1f}%")
    
    # Model configuration
    config = model.config
    print(f"\n⚙️  Model Configuration:")
    print(f"   Architecture: {config.n_layer} layers, {config.n_embd} dim, {config.n_head} heads")
    print(f"   MoE: {config.num_experts} experts, {config.experts_per_token} per token")
    print(f"   Shared weight ratio: {config.shared_weight_ratio*100:.1f}%")
    print(f"   Block size: {config.block_size}")
    print(f"   Vocab size: {config.vocab_size}")
    print(f"   Using Dynamic Tanh: {config.use_dyt}")
    print(f"   Using FP8: {config.use_fp8}")
    
    # Test forward pass
    print(f"\n🧪 Forward Pass Test:")
    forward_test_passed = test_model_forward(model, device)
    
    # Memory usage estimation
    print(f"\n💾 Memory Usage Estimation:")
    param_memory_mb = total_params * 4 / (1024 * 1024)  # FP32
    param_memory_mb_fp16 = total_params * 2 / (1024 * 1024)  # FP16/BF16
    
    # Estimate activation memory for a typical batch
    batch_size, seq_len = 16, 2048
    activation_memory_mb = estimate_activation_memory(model.config, batch_size, seq_len)
    
    print(f"   Parameters (FP32): {param_memory_mb:.1f} MB")
    print(f"   Parameters (FP16/BF16): {param_memory_mb_fp16:.1f} MB")
    print(f"   Estimated activations (batch={batch_size}, seq={seq_len}): {activation_memory_mb:.1f} MB")
    print(f"   Total training memory (FP16): {param_memory_mb_fp16 + activation_memory_mb:.1f} MB")
    
    # Overall assessment
    print(f"\n" + "=" * 80)
    print("📋 FINAL ASSESSMENT")
    print("=" * 80)
    
    all_tests_passed = within_tolerance and forward_test_passed
    
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print(f"   ✅ Parameter count within tolerance")
        print(f"   ✅ Forward pass successful")
        print(f"   ✅ Model architecture validated")
        print(f"\nThe SLM-MoE-MLA model is ready for training!")
    else:
        print("⚠️  SOME TESTS FAILED")
        if not within_tolerance:
            print(f"   ❌ Parameter count outside tolerance")
        if not forward_test_passed:
            print(f"   ❌ Forward pass failed")
        print(f"\nPlease review the model configuration.")
    
    return all_tests_passed

def estimate_activation_memory(config, batch_size: int, seq_len: int) -> float:
    """Estimate activation memory usage in MB."""
    # Rough estimation of activation memory
    # This is a simplified calculation
    
    n_embd = config.n_embd
    n_layer = config.n_layer
    vocab_size = config.vocab_size
    
    # Embeddings
    embedding_memory = batch_size * seq_len * n_embd * 2  # bytes (FP16)
    
    # Attention activations per layer
    # Q, K, V projections + attention scores + output
    attn_memory = batch_size * seq_len * n_embd * 4 * n_layer * 2
    
    # MLP activations per layer (considering MoE routing)
    mlp_memory = batch_size * seq_len * config.n_inner * config.experts_per_token / config.num_experts * n_layer * 2
    
    # Output head
    output_memory = batch_size * seq_len * vocab_size * 2
    
    total_bytes = embedding_memory + attn_memory + mlp_memory + output_memory
    return total_bytes / (1024 * 1024)  # Convert to MB

def main():
    parser = argparse.ArgumentParser(description='Test SLM-MoE-MLA parameter count and functionality')
    parser.add_argument('--size', type=str, default='small', choices=['tiny', 'small', 'medium', 'large'],
                       help='Model size to test')
    parser.add_argument('--target_params', type=float, default=200.0,
                       help='Target parameter count in millions')
    parser.add_argument('--tolerance', type=float, default=0.1,
                       help='Tolerance for parameter count (as fraction of target)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run tests on (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Adjust target params based on size
    if args.size == 'tiny':
        target_params = 100.0
    elif args.size == 'small':
        target_params = 200.0
    elif args.size == 'medium':
        target_params = 400.0
    elif args.size == 'large':
        target_params = 800.0
    else:
        target_params = args.target_params
    
    success = run_parameter_tests(
        model_size=args.size,
        target_params_m=target_params,
        tolerance=args.tolerance,
        device=args.device
    )
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()