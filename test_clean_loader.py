#!/usr/bin/env python3
"""Test the dynamic loader with cleaner logging"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', '')

import time
from data.data_loader_dynamic import DynamicFinewebDataset

def main():
    print("\nTesting Dynamic Data Loader with Clean Logging")
    print("=" * 60)
    
    # Create dataset
    dataset = DynamicFinewebDataset(
        max_sequences_per_batch=4,
        buffer_size=16,
        max_length=512,
        num_tokenizer_workers=2,
        prefetch_size=100,
        gradient_accumulation_steps=2
    )
    
    print("\nFetching batches...")
    start = time.time()
    
    # Get multiple batches
    batch_count = 20
    for i in range(batch_count):
        batch = next(iter(dataset))
        if i == 0:
            print(f"First batch shape: {batch['input_ids'].shape}")
            print(f"Batch keys: {list(batch.keys())}")
    
    elapsed = time.time() - start
    print(f"\nProcessed {batch_count} batches in {elapsed:.2f}s")
    print(f"Throughput: {batch_count/elapsed:.1f} batches/sec")
    
    # Get final statistics
    stats = dataset.get_stats()
    print(f"\nPerformance Summary:")
    print(f"  Documents tokenized: {stats['docs_tokenized']:,}")
    print(f"  Batches created: {stats['batches_created']:,}")
    print(f"  Batches served: {stats['batches_served']:,}")
    print(f"  Padding efficiency: {(1 - stats['avg_padding_ratio']) * 100:.1f}%")
    
    dataset.close()
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()