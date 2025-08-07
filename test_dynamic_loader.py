#!/usr/bin/env python3
"""Test script for the dynamic data loader with mock data"""

import torch
from typing import Dict, List
import time
from datetime import datetime

class MockDataset:
    """Mock dataset for testing without external dependencies"""
    def __init__(self, num_docs=1000):
        self.num_docs = num_docs
        self.current = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current >= self.num_docs:
            raise StopIteration
        self.current += 1
        # Generate random text of varying lengths
        import random
        length = random.randint(100, 1000)
        text = f"Sample document {self.current} with " + "text " * length
        return {'text': text}
    
    def skip(self, n):
        self.current += n
        return self

class MockTokenizer:
    """Mock tokenizer for testing"""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token = "[EOS]"
        self.pad_token = "[PAD]"
    
    def __call__(self, text, **kwargs):
        # Simple mock tokenization
        import random
        if isinstance(text, str):
            length = min(len(text.split()), kwargs.get('max_length', 512))
            length = random.randint(length//2, length)  # Variable lengths
            input_ids = torch.randint(1, 1000, (1, length))
        else:
            # Batch tokenization
            lengths = [min(len(t.split()), kwargs.get('max_length', 512)) for t in text]
            max_len = max(lengths)
            input_ids = torch.zeros((len(text), max_len), dtype=torch.long)
            for i, l in enumerate(lengths):
                input_ids[i, :l] = torch.randint(1, 1000, (l,))
        
        return {'input_ids': input_ids, 'attention_mask': (input_ids != 0).long()}

def test_dynamic_loader():
    """Test the dynamic loader with mock components"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting test with mock data")
    
    # Monkey-patch the imports in the module
    import sys
    import data.data_loader_dynamic as loader_module
    
    # Replace load_dataset with our mock
    original_load_dataset = loader_module.load_dataset
    loader_module.load_dataset = lambda *args, **kwargs: MockDataset()
    
    try:
        # Create the dataset with mock tokenizer
        dataset = loader_module.DynamicFinewebDataset(
            max_sequences_per_batch=4,
            buffer_size=8,
            max_length=256,
            gradient_accumulation_steps=2,
            num_tokenizer_workers=2,
            prefetch_size=20,
            tokenizer=MockTokenizer()
        )
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Dataset initialized, fetching batches...")
        
        # Get a few batches
        for i in range(3):
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Fetching batch {i+1}")
            batch = next(iter(dataset))
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Got batch with shape: {batch['input_ids'].shape}")
            print(f"  Keys: {list(batch.keys())}")
        
        # Get statistics
        stats = dataset.get_stats()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Statistics:")
        print(f"  Docs tokenized: {stats['docs_tokenized']}")
        print(f"  Batches created: {stats['batches_created']}")
        print(f"  Batches served: {stats['batches_served']}")
        print(f"  Padding ratio: {stats['avg_padding_ratio']:.2%}")
        
        dataset.close()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Test completed successfully!")
        
    finally:
        # Restore original
        loader_module.load_dataset = original_load_dataset

if __name__ == "__main__":
    test_dynamic_loader()