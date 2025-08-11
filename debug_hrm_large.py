#!/usr/bin/env python3
"""Debug HRM model with actual vocab size"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.models.hrm_model import HRM, HRMConfig

# Create model with actual vocab size but simple config
config = HRMConfig(
    n_layer=1,
    n_embd=256,
    n_head=4,
    n_inner=512,
    vocab_size=128256,  # Actual vocab size
    block_size=128,
    cycles_per_segment=2,
    steps_per_cycle=3,
    max_segments=2,
    use_act=False,  # Disable ACT for now
    use_deep_supervision=False,
    dropout=0.0
)

model = HRM(config)
print(f"Model created with {model.param_count/1e6:.2f}M parameters")
print(f"Vocab size: {config.vocab_size}")

# Test with simple input
batch_size = 2
seq_len = 32
input_ids = torch.randint(0, 1000, (batch_size, seq_len))  # Use only first 1000 tokens
labels = torch.randint(0, 1000, (batch_size, seq_len))

# Forward pass
model.eval()
with torch.no_grad():
    outputs = model(input_ids, labels=labels)

print(f"Loss: {outputs['loss'].item():.4f}")

# Expected loss
expected_loss = torch.log(torch.tensor(float(config.vocab_size)))
print(f"Expected initial loss (random): ~{expected_loss.item():.2f}")

# Check logits statistics
logits = outputs['logits']
print(f"Logits mean: {logits.mean().item():.6f}")
print(f"Logits std: {logits.std().item():.6f}")
print(f"Logits min: {logits.min().item():.6f}")
print(f"Logits max: {logits.max().item():.6f}")

# Check if there are NaN or Inf values
print(f"Logits has NaN: {torch.isnan(logits).any().item()}")
print(f"Logits has Inf: {torch.isinf(logits).any().item()}")

# Test the embeddings alone
print("\n--- Testing embeddings ---")
token_emb = model.token_embeddings(input_ids)
pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
pos_emb = model.pos_embeddings(pos_ids)
x_embedded = model.dropout(token_emb + pos_emb)

print(f"Token embeddings mean: {token_emb.mean().item():.6f}")
print(f"Token embeddings std: {token_emb.std().item():.6f}")
print(f"Pos embeddings mean: {pos_emb.mean().item():.6f}")
print(f"Pos embeddings std: {pos_emb.std().item():.6f}")

# Check the lm_head weights
print("\n--- Checking lm_head ---")
print(f"lm_head weight shape: {model.lm_head.weight.shape}")
print(f"lm_head weight mean: {model.lm_head.weight.mean().item():.6f}")
print(f"lm_head weight std: {model.lm_head.weight.std().item():.6f}")