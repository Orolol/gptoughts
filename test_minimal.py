#!/usr/bin/env python3
import os
import sys
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("Starting minimal test...")
sys.stdout.flush()

from data.data_loader_dynamic import DynamicFinewebDataset

print("Creating dataset...")
sys.stdout.flush()

try:
    dataset = DynamicFinewebDataset(
        max_sequences_per_batch=2,
        buffer_size=4,
        max_length=128,
        num_tokenizer_workers=1,
        prefetch_size=10,
        gradient_accumulation_steps=1
    )
    print("Dataset created!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Test complete")
dataset.close()