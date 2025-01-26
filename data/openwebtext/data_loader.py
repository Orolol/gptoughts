import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import pickle
import time

class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
    
    def update(self, new_tokens):
        self.total_tokens += new_tokens

class StreamingDataset(IterableDataset):
    def __init__(self, block_size, batch_size, dataset_name="HuggingFaceFW/fineweb-edu", 
                 dataset_config="CC-MAIN-2024-10", split="train", device='cuda'):
        super().__init__()
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.dataset_config = dataset_config
        self.split = split
        
        # Add prefetch parameters
        self.prefetch_factor = 4  # Number of batches to prefetch
        self.pin_memory = True
        self.num_workers = min(4, os.cpu_count() // 2)  # Optimal worker count
        
        access_token = os.getenv('HF_TOKEN')
        
        # Initialize tokenizer with faster options
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct", 
            use_fast=True, 
            access_token=access_token,
            padding_side='left',  # Optimize for left-padded sequences
            truncation_side='right'
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize dataset with better streaming params
        self.dataset = load_dataset(dataset_name, name=dataset_config, split=split, 
                                  streaming=True)
        self.dataset = self.dataset.with_format("torch")
        self.dataset = self.dataset.shuffle(buffer_size=10_000)
        
        # Create dataloader and iterator
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size * 4,  # Pre-batch for efficiency
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )
        self._iterator = iter(self.dataloader)
        
        # Preallocate buffers
        self.token_buffer = torch.empty((batch_size * block_size * 8), 
                                      dtype=torch.long, 
                                      device='cpu',
                                      pin_memory=True)
        self.ptr = 0
        
        # Initialize token tracker
        tracker_dir = os.path.join('data', 'token_tracking')
        os.makedirs(tracker_dir, exist_ok=True)
        self.token_tracker = TokenTracker()
        
        # Save tokenizer metadata
        self.save_meta()
    
    def save_meta(self):
        meta = {
            'vocab_size': len(self.tokenizer),
            'tokenizer_name': "meta-llama/Llama-3.2-1B-Instruct",
            'tokenizer': self.tokenizer
        }
        meta_dir = os.path.join('data', 'openwebtext')
        os.makedirs(meta_dir, exist_ok=True)
        with open(os.path.join(meta_dir, 'meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)
    
    def process_example(self, example):
        """Optimized tokenization with memory reuse"""
        # Reuse existing buffer to avoid allocations
        ids = self.tokenizer.encode_plus(
            example['text'],
            add_special_tokens=False,
            return_tensors='pt',
            max_length=self.block_size,
            truncation=True,
            padding='max_length',
            return_attention_mask=False
        ).input_ids.squeeze(0)
        
        ids = ids.to(device='cpu', non_blocking=True)
        return ids
    
    def get_batch(self):
        """Vectorized batch preparation"""
        # Async prefetch next batch while processing
        with torch.cuda.stream(torch.cuda.Stream()):
            # Fill buffer in chunks
            while self.ptr < self.batch_size * self.block_size * 2:
                try:
                    batch = next(self._iterator)
                except StopIteration:
                    # Reset iterator if exhausted
                    self._iterator = iter(self.dataloader)
                    batch = next(self._iterator)
                
                tokens = self.process_example(batch)
                
                # Copy to buffer
                avail = min(tokens.numel(), self.token_buffer.numel() - self.ptr)
                self.token_buffer[self.ptr:self.ptr+avail] = tokens[:avail]
                self.ptr += avail
            
            # Prepare tensors with async copy
            encoder_input = self.token_buffer[:self.batch_size*self.block_size]\
                .view(self.batch_size, self.block_size)\
                .to(device=self.device, non_blocking=True)
            
            decoder_input = self.token_buffer[self.batch_size*self.block_size:self.batch_size*self.block_size*2]\
                .view(self.batch_size, self.block_size)\
                .to(device=self.device, non_blocking=True)
            
            # Create target by shifting decoder input
            target = decoder_input.clone()
            target = torch.roll(target, shifts=-1, dims=1)
            target[:, -1] = -1  # Masquer le dernier token
            
            # Rotate buffer
            self.token_buffer = torch.roll(self.token_buffer, -self.batch_size*self.block_size*2)
            self.ptr -= self.batch_size*self.block_size*2
            
            # Update token count
            self.token_tracker.update(self.batch_size * self.block_size * 2)
        
        return encoder_input, decoder_input, target

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.get_batch()
    
    def get_total_tokens(self):
        """Return the total number of tokens processed so far."""
        return self.token_tracker.total_tokens

# Example usage:
if __name__ == "__main__":
    # Create dataset with block size of 1024 and batch size of 4
    dataset = StreamingDataset(block_size=1024, batch_size=4)
    
    # Get a few batches
    for i, (encoder_input, decoder_input, target) in enumerate(dataset):
        print(f"Batch {i}:")
        print(f"Encoder input shape: {encoder_input.shape}")
        print(f"Decoder input shape: {decoder_input.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Total tokens processed: {dataset.get_total_tokens():,}")
        
        if i >= 2:  # Just show first 3 batches
            break
