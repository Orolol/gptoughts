#!/usr/bin/env python3
"""Test script to verify proper cleanup on interrupt"""

import time
import torch
from data.data_loader_dynamic import DynamicFinewebDataset

def main():
    print("Starting interrupt test...")
    print("Press Ctrl+C to test interrupt handling")
    
    # Create dataset
    dataset = DynamicFinewebDataset(
        max_sequences_per_batch=4,
        buffer_size=8,
        max_length=512,
        num_tokenizer_workers=2,
        prefetch_size=100,
        gradient_accumulation_steps=2
    )
    
    try:
        # Start iterating
        print("\nStarting iteration, press Ctrl+C to interrupt...")
        for i, batch in enumerate(dataset):
            print(f"Batch {i}: shape={batch['input_ids'].shape}")
            time.sleep(0.5)  # Slow down for testing
            
            if i >= 10:
                print("Reached 10 batches, stopping normally...")
                break
                
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught - testing cleanup...")
    finally:
        print("\nCalling close() explicitly...")
        dataset.close()
        print("Test complete!")

if __name__ == "__main__":
    main()