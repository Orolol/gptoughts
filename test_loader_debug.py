#!/usr/bin/env python3
"""Debug test for the dynamic loader"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', '')

import sys
import time
from datetime import datetime

print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting debug test")

# Import the loader
print(f"[{datetime.now().strftime('%H:%M:%S')}] Importing data loader...")
from data.data_loader_dynamic import DynamicFinewebDataset
print(f"[{datetime.now().strftime('%H:%M:%S')}] Import successful")

# Create dataset with minimal config
print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating dataset...")
try:
    dataset = DynamicFinewebDataset(
        max_sequences_per_batch=2,
        buffer_size=2,
        max_length=128,
        num_tokenizer_workers=1,
        prefetch_size=4,
        gradient_accumulation_steps=1
    )
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset created successfully")
except Exception as e:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR creating dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Test complete!")