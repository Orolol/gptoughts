import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset
import numpy as np
import pickle

class TokenTracker:
    def __init__(self, save_path='token_count.json'):
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.total_tokens = self.load_count()
    
    def load_count(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                return json.load(f)['total_tokens']
        return 0
    
    def save_count(self):
        with open(self.save_path, 'w') as f:
            json.dump({'total_tokens': self.total_tokens}, f)
    
    def update(self, new_tokens):
        self.total_tokens += new_tokens
        self.save_count()

class StreamingDataset(IterableDataset):
    def __init__(self, block_size, batch_size, dataset_name="HuggingFaceFW/fineweb-2", 
                 dataset_config="fra_Latn", split="train", device='cuda'):
        super().__init__()
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.dataset_config = dataset_config
        self.split = split
        
        access_token = os.getenv('HF_TOKEN')
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_fast=True, access_token=access_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize dataset
        self.dataset = load_dataset(dataset_name, name=dataset_config, split=split, streaming=True)
        
        # Initialize token tracker
        tracker_dir = os.path.join('data', 'token_tracking')
        os.makedirs(tracker_dir, exist_ok=True)
        self.token_tracker = TokenTracker(os.path.join(tracker_dir, f'{split}_token_count.json'))
        
        # Buffer for storing tokens
        self.token_buffer = []
        
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
        """Tokenize a single example."""
        ids = self.tokenizer.encode(example['text'], add_special_tokens=False)
        ids.append(self.tokenizer.eos_token_id)
        return ids
    
    def get_batch(self):
        """Get a batch of token sequences of length block_size."""
        while len(self.token_buffer) < (self.block_size * self.batch_size * 2 + 1):  # Double the buffer size for encoder+decoder
            try:
                example = next(iter(self.dataset))
                new_tokens = self.process_example(example)
                self.token_buffer.extend(new_tokens)
                self.token_tracker.update(len(new_tokens))
            except StopIteration:
                self.dataset = load_dataset("HuggingFaceFW/fineweb-2", 
                                         name=self.dataset_config,  # Use the instance variable 
                                         split=self.split,         # Use the instance variable
                                         streaming=True)
        
        # Split the sequence into encoder input and decoder input
        total_length = self.block_size * self.batch_size
        
        # Prepare encoder input
        encoder_input = torch.tensor(self.token_buffer[total_length:total_length*2], 
                                   dtype=torch.long, device=self.device)
        
        # Prepare decoder input (shifted by 1 relative to target)
        decoder_input = torch.tensor(self.token_buffer[total_length:total_length*2], 
                                   dtype=torch.long, device=self.device)
        
        # Prepare target (shifted by 1 relative to decoder input)
        target = torch.tensor(self.token_buffer[total_length+1:total_length*2+1], 
                             dtype=torch.long, device=self.device)
        
        # Reshape into batch_size x block_size
        encoder_input = encoder_input.view(self.batch_size, self.block_size)
        decoder_input = decoder_input.view(self.batch_size, self.block_size)
        target = target.view(self.batch_size, self.block_size)
        
        # Remove used tokens
        self.token_buffer = self.token_buffer[total_length*2:]
        
        return encoder_input, decoder_input, target
    
    def __iter__(self):
        while True:
            yield self.get_batch()
    
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