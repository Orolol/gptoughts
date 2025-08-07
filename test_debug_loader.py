#!/usr/bin/env python3
"""Debug test for the data loader timeout issue"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("Starting debug test...")

from data.data_loader_dynamic import DynamicFinewebDataset

print("Creating dataset with minimal config...")
dataset = DynamicFinewebDataset(
    max_sequences_per_batch=2,
    buffer_size=4,
    max_length=128,
    num_tokenizer_workers=1,
    prefetch_size=10,
    gradient_accumulation_steps=1
)

print("Dataset created!")
print(f"Stats: {dataset.get_stats()}")

# Try to get a batch
print("Attempting to get a batch...")
try:
    batch = next(iter(dataset))
    print(f"Got batch: {batch['input_ids'].shape}")
except Exception as e:
    print(f"Error getting batch: {e}")
    import traceback
    traceback.print_exc()

dataset.close()
print("Test complete")