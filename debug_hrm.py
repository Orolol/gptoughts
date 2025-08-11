#!/usr/bin/env python3
"""Debug HRM model to find issue with high loss"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.models.hrm_model import HRM, HRMConfig

# Create a minimal model
config = HRMConfig(
    n_layer=1,
    n_embd=128,
    n_head=4,
    n_inner=256,
    vocab_size=100,  # Very small vocab
    block_size=32,
    cycles_per_segment=1,
    steps_per_cycle=1,
    max_segments=1,
    use_act=False,  # Disable ACT
    use_deep_supervision=False,  # Disable deep supervision
    dropout=0.0
)

model = HRM(config)
print(f"Model created with {model.param_count/1e6:.2f}M parameters")

# Test with very simple input
batch_size = 1
seq_len = 16
input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
labels = torch.ones((batch_size, seq_len), dtype=torch.long)  # All same label

# Forward pass
model.eval()  # Use eval mode to avoid training-specific paths
with torch.no_grad():
    outputs = model(input_ids, labels=labels)

print(f"Loss: {outputs['loss'].item():.4f}")
print(f"Logits shape: {outputs['logits'].shape}")

# Check logits statistics
logits = outputs['logits']
print(f"Logits mean: {logits.mean().item():.4f}")
print(f"Logits std: {logits.std().item():.4f}")
print(f"Logits min: {logits.min().item():.4f}")
print(f"Logits max: {logits.max().item():.4f}")

# Calculate loss manually
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
manual_loss = F.cross_entropy(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1))
print(f"Manual loss: {manual_loss.item():.4f}")

# Expected loss for random initialization
expected_loss = torch.log(torch.tensor(float(config.vocab_size)))
print(f"Expected initial loss (random): ~{expected_loss.item():.2f}")

# Test individual components
print("\n--- Testing components ---")

# Test embeddings
x_embedded = model.compute_embeddings(input_ids)
print(f"Embeddings shape: {x_embedded.shape}")
print(f"Embeddings mean: {x_embedded.mean().item():.4f}")
print(f"Embeddings std: {x_embedded.std().item():.4f}")

# Test initial states
z_L = torch.zeros_like(x_embedded)
z_H = torch.zeros_like(x_embedded)

# Test one forward segment
z_H_out, z_L_out, conv_metrics = model.forward_segment(z_H, z_L, x_embedded)
print(f"z_H output mean: {z_H_out.mean().item():.4f}")
print(f"z_H output std: {z_H_out.std().item():.4f}")
print(f"z_L output mean: {z_L_out.mean().item():.4f}")
print(f"z_L output std: {z_L_out.std().item():.4f}")

# Test lm_head
final_logits = model.lm_head(z_H_out)
print(f"Final logits mean: {final_logits.mean().item():.4f}")
print(f"Final logits std: {final_logits.std().item():.4f}")