#!/usr/bin/env python3
"""Complete test for the dynamic loader with batch fetching"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', '')

import sys
import time
from datetime import datetime

print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting complete test")

# Import the loader
print(f"[{datetime.now().strftime('%H:%M:%S')}] Importing data loader...")
from data.data_loader_dynamic import DynamicFinewebDataset
print(f"[{datetime.now().strftime('%H:%M:%S')}] Import successful")

# Create dataset with typical config
print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating dataset...")
try:
    dataset = DynamicFinewebDataset(
        max_sequences_per_batch=4,
        buffer_size=8,
        max_length=512,
        num_tokenizer_workers=2,
        prefetch_size=50,
        gradient_accumulation_steps=2
    )
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset created successfully")
except Exception as e:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR creating dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Fetch some batches
print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching batches...")
try:
    for i in range(5):
        batch = next(iter(dataset))
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Batch {i+1}: shape={batch['input_ids'].shape}")
        
    # Get stats
    stats = dataset.get_stats()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Statistics:")
    print(f"  - Docs tokenized: {stats['docs_tokenized']:,}")
    print(f"  - Batches created: {stats['batches_created']}")
    print(f"  - Batches served: {stats['batches_served']}")
    print(f"  - Padding efficiency: {(1 - stats['avg_padding_ratio']) * 100:.1f}%")
    
    dataset.close()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Test complete!")
    
except Exception as e:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR during batch fetching: {e}")
    import traceback
    traceback.print_exc()
    dataset.close()
    sys.exit(1)