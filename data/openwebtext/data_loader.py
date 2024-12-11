import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset
import numpy as np
import pickle

class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
    
    def update(self, new_tokens):
        self.total_tokens += new_tokens

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
        self.token_tracker = TokenTracker()
        
        # Modifions la façon dont nous gérons le dataset
        self.dataset_iterator = iter(self.dataset)
        self.prefetch_factor = 10000  # Augmenter la taille du buffer
        self.token_buffer = []
        
        # Ajouter le pin memory pour des transferts plus rapides
        self.pin_memory = device == 'cuda'
        
        # Précharger le premier batch
        self._prefetch_tokens()
        
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
    
    def _prefetch_tokens(self):
        """Précharge un grand nombre de tokens en une fois."""
        target_size = self.prefetch_factor * self.block_size * self.batch_size
        
        while len(self.token_buffer) < target_size:
            try:
                examples = [next(self.dataset_iterator) for _ in range(100)]  # Batch processing
                tokens = []
                for example in examples:
                    tokens.extend(self.process_example(example))
                self.token_buffer.extend(tokens)
                self.token_tracker.update(len(tokens))
            except StopIteration:
                self.dataset_iterator = iter(self.dataset)
                if len(self.token_buffer) == 0:
                    continue
                break
    
    def get_batch(self):
        """Get a batch of token sequences of length block_size."""
        if len(self.token_buffer) < (self.block_size * self.batch_size * 2 + 1):
            self._prefetch_tokens()
            
        total_length = self.block_size * self.batch_size
        
        # Créer les tensors directement sur le device
        encoder_input = torch.tensor(
            self.token_buffer[:total_length], 
            dtype=torch.long, 
            device=self.device,
            pin_memory=self.pin_memory
        ).view(self.batch_size, self.block_size)
        
        decoder_input = torch.tensor(
            self.token_buffer[total_length:total_length*2], 
            dtype=torch.long, 
            device=self.device,
            pin_memory=self.pin_memory
        ).view(self.batch_size, self.block_size)
        
        target = torch.tensor(
            self.token_buffer[total_length+1:total_length*2+1], 
            dtype=torch.long, 
            device=self.device,
            pin_memory=self.pin_memory
        ).view(self.batch_size, self.block_size)
        
        # Nettoyer le buffer
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