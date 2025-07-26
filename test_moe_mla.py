"""Quick test for MOE-MLA model initialization and performance"""

import torch
import time
import subprocess
from models.models.moe_mla_model import create_moe_mla_model, MOEMLAConfig

# Test model creation
print("Testing MOE-MLA model creation...")

# Create a small model
config = MOEMLAConfig(
    n_layer=4,
    n_embd=256,
    n_head=4,
    vocab_size=32000,
    block_size=512,
    num_experts=4,
    experts_per_token=2,
    shared_weight_ratio=0.75,
    use_fp8=False,
    use_dyt=True
)

model = create_moe_mla_model(
    size='small',
    num_experts=4,
    experts_per_token=2,
    block_size=512,
    use_fp8=False,
    use_dyt=True
)

print("Model created successfully!")
print(f"Parameter count: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# Test forward pass
batch_size = 2
seq_len = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = model.to(device)
input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
targets = input_ids.clone()

print(f"\nTesting forward pass on {device}...")
with torch.cuda.amp.autocast(enabled=True):
    logits, loss = model(input_ids, targets)

print(f"Forward pass successful!")
print(f"Logits shape: {logits.shape}")
print(f"Loss: {loss.item():.4f}")

# Test GaLore2 optimizer
print("\nTesting GaLore2 optimizer...")
from models.galore2_fixed import GaLore2AdamW

optimizer = GaLore2AdamW(
    model.parameters(),
    lr=1e-4,
    rank=64,
    update_proj_gap=10,
    scale=0.25
)

# Test optimization step
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Optimization step successful!")

# Performance test
print("\n" + "="*50)
print("PERFORMANCE TEST")
print("="*50)

# Create a larger model for performance testing
print("\nCreating larger model for performance test...")
perf_model = create_moe_mla_model(
    size='medium',
    num_experts=8,
    experts_per_token=2,
    block_size=2048,
    use_fp8=False,
    use_dyt=True
).to(device)

print(f"Performance test model: {sum(p.numel() for p in perf_model.parameters()) / 1e6:.2f}M parameters")

# Larger batch for performance testing
batch_size = 4
seq_len = 1024

print(f"\nRunning performance test with batch_size={batch_size}, seq_len={seq_len}")
print("Monitoring GPU utilization...")

# Function to get GPU utilization
def get_gpu_utilization():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        return int(result.stdout.strip())
    except:
        return None

# Warmup
print("\nWarmup phase...")
for _ in range(3):
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    targets = input_ids.clone()
    with torch.cuda.amp.autocast(enabled=True):
        logits, loss = perf_model(input_ids, targets)
    loss.backward()
    torch.cuda.synchronize()

# Performance measurement
print("\nMeasuring performance...")
gpu_utilizations = []
forward_times = []
backward_times = []

num_iterations = 10
for i in range(num_iterations):
    # Get initial GPU utilization
    gpu_util = get_gpu_utilization()
    if gpu_util is not None:
        gpu_utilizations.append(gpu_util)
    
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    targets = input_ids.clone()
    
    # Forward pass timing
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.cuda.amp.autocast(enabled=True):
        logits, loss = perf_model(input_ids, targets)
    
    torch.cuda.synchronize()
    forward_time = time.time() - start_time
    forward_times.append(forward_time)
    
    # Backward pass timing
    start_time = time.time()
    loss.backward()
    torch.cuda.synchronize()
    backward_time = time.time() - start_time
    backward_times.append(backward_time)
    
    if i % 2 == 0:
        print(f"Iteration {i+1}/{num_iterations}: Forward: {forward_time:.3f}s, Backward: {backward_time:.3f}s")

# Calculate statistics
avg_forward = sum(forward_times) / len(forward_times)
avg_backward = sum(backward_times) / len(backward_times)
avg_total = avg_forward + avg_backward

print("\n" + "="*50)
print("PERFORMANCE RESULTS")
print("="*50)
print(f"Average forward pass time: {avg_forward:.3f}s")
print(f"Average backward pass time: {avg_backward:.3f}s")
print(f"Average total time per iteration: {avg_total:.3f}s")
print(f"Throughput: {batch_size * seq_len / avg_total:.0f} tokens/second")

if gpu_utilizations:
    avg_gpu_util = sum(gpu_utilizations) / len(gpu_utilizations)
    max_gpu_util = max(gpu_utilizations)
    print(f"\nAverage GPU utilization: {avg_gpu_util:.0f}%")
    print(f"Peak GPU utilization: {max_gpu_util:.0f}%")
    if avg_gpu_util < 80:
        print("⚠️  WARNING: Low GPU utilization detected!")
    else:
        print("✅ Good GPU utilization!")

# Estimate MFU
print(f"\nEstimating Model FLOPs Utilization (MFU)...")
mfu = perf_model.estimate_mfu(batch_size, avg_total, seq_len)
print(f"MFU: {mfu:.1f}%")

print("\nAll tests passed!")