#!/usr/bin/env python3
"""
Test script to verify MLA model memory management improvements.
This tests that the tensor lifecycle fixes prevent memory leaks during training.
"""

import torch
import torch.nn as nn
import gc
from models.models.mla_model import create_mla_model, MLAModelConfig

def format_memory(bytes):
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"

def get_memory_stats():
    """Get current GPU memory statistics."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        return {
            'allocated': allocated,
            'reserved': reserved,
            'allocated_str': format_memory(allocated),
            'reserved_str': format_memory(reserved)
        }
    return None

def test_memory_lifecycle():
    """Test that MLA model doesn't leak memory during training."""
    print("Testing MLA Model Memory Lifecycle...")
    print("=" * 60)
    
    # Create a small model for testing
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = create_mla_model(
        size='small',
        vocab_size=1000,  # Small vocab for testing
        block_size=512,   # Small sequence length
        dropout=0.0,
        fp8_params=False  # Disable FP8 for testing
    ).to(device)
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Initial memory state
    torch.cuda.empty_cache()
    gc.collect()
    initial_memory = get_memory_stats()
    print(f"\nInitial memory: {initial_memory['allocated_str']} allocated, {initial_memory['reserved_str']} reserved")
    
    # Simulate training steps
    batch_size = 4
    seq_length = 512
    num_steps = 10
    
    memory_history = []
    
    print(f"\nRunning {num_steps} training steps...")
    print("-" * 60)
    
    for step in range(num_steps):
        # Create random input
        input_ids = torch.randint(0, 1000, (batch_size, seq_length), device=device)
        targets = torch.randint(0, 1000, (batch_size, seq_length), device=device)
        
        # Forward pass
        logits, loss = model(input_ids, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Get memory stats after step
        mem_stats = get_memory_stats()
        memory_history.append(mem_stats['allocated'])
        
        if step % 2 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}, "
                  f"Memory = {mem_stats['allocated_str']} allocated, "
                  f"{mem_stats['reserved_str']} reserved")
    
    # Final memory state
    torch.cuda.empty_cache()
    gc.collect()
    final_memory = get_memory_stats()
    print("-" * 60)
    print(f"\nFinal memory: {final_memory['allocated_str']} allocated, {final_memory['reserved_str']} reserved")
    
    # Check for memory leak
    memory_increase = final_memory['allocated'] - initial_memory['allocated']
    memory_increase_pct = (memory_increase / initial_memory['allocated']) * 100 if initial_memory['allocated'] > 0 else 0
    
    print(f"\nMemory increase: {format_memory(memory_increase)} ({memory_increase_pct:.1f}%)")
    
    # Analyze memory trend
    if len(memory_history) > 2:
        # Check if memory is steadily increasing
        steady_increase = all(memory_history[i] <= memory_history[i+1] for i in range(len(memory_history)-1))
        
        if steady_increase and memory_increase_pct > 20:
            print("\n⚠️  WARNING: Potential memory leak detected!")
            print("   Memory usage increased steadily during training.")
        else:
            print("\n✅ SUCCESS: No significant memory leak detected!")
            print("   Memory usage appears stable during training.")
    
    # Test inference mode
    print("\n" + "=" * 60)
    print("Testing inference mode memory usage...")
    
    model.eval()
    with torch.no_grad():
        # Test generation
        prompt = torch.randint(0, 1000, (1, 10), device=device)
        print(f"\nGenerating 20 tokens...")
        
        # Measure memory before generation
        torch.cuda.empty_cache()
        before_gen = get_memory_stats()
        
        # Generate
        output, _ = model.generate(prompt, max_new_tokens=20, temperature=1.0)
        
        # Measure memory after generation
        after_gen = get_memory_stats()
        
        gen_memory_increase = after_gen['allocated'] - before_gen['allocated']
        print(f"Memory used for generation: {format_memory(gen_memory_increase)}")
        
        # Clear caches
        model.clear_cache()
        torch.cuda.empty_cache()
        
        after_clear = get_memory_stats()
        memory_freed = after_gen['allocated'] - after_clear['allocated']
        print(f"Memory freed after cache clear: {format_memory(memory_freed)}")
    
    print("\n" + "=" * 60)
    print("Memory lifecycle test completed!")

if __name__ == "__main__":
    test_memory_lifecycle()