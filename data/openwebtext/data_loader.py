import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset
import numpy as np
import pickle
import itertools

class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
    
    def update(self, new_tokens):
        self.total_tokens += new_tokens

class StreamingDataset(IterableDataset):
    def __init__(self, block_size, batch_size, dataset_name="HuggingFaceFW/fineweb-2", 
                 dataset_config="fra_Latn", split="train", device='cuda',
                 buffer_size=10000):
        super().__init__()
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.buffer_size = buffer_size
        
        access_token = os.getenv('HF_TOKEN')
        
        # Initialiser le tokenizer avec fast tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct", 
            use_fast=True, 
            access_token=access_token,
            model_max_length=block_size
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Charger le dataset en mode streaming
        self.dataset = load_dataset(
            dataset_name, 
            name=dataset_config, 
            split=split, 
            streaming=True
        ).shuffle(buffer_size=1000)
        
        # Préallouer le buffer
        self.token_buffer = []
        self.prefetch_buffer = []
        
        # Initialiser le token tracker
        self.token_tracker = TokenTracker()
        
    def process_example(self, example):
        """Tokenize un exemple avec batch processing"""
        return self.tokenizer(
            example['text'],
            add_special_tokens=False,
            truncation=True,
            max_length=self.block_size,
            return_tensors='pt'
        )['input_ids'].squeeze(0)

    def get_batch(self):
        """Optimisation du get_batch avec préchargement"""
        # Remplir le buffer si nécessaire
        while len(self.token_buffer) < (self.block_size * self.batch_size * 2 + 1):
            if not self.prefetch_buffer:
                # Précharger plusieurs exemples d'un coup
                examples = list(itertools.islice(self.dataset, 32))
                if not examples:
                    self.dataset = load_dataset(
                        "HuggingFaceFW/fineweb-2",
                        name=self.dataset_config,
                        split=self.split,
                        streaming=True
                    ).shuffle(buffer_size=1000)
                    examples = list(itertools.islice(self.dataset, 32))
                
                # Tokenizer en batch
                tokenized = [self.process_example(ex) for ex in examples]
                self.prefetch_buffer.extend([t for tokens in tokenized for t in tokens])
                
            # Transférer du prefetch buffer au token buffer
            transfer_size = min(self.buffer_size, len(self.prefetch_buffer))
            self.token_buffer.extend(self.prefetch_buffer[:transfer_size])
            self.prefetch_buffer = self.prefetch_buffer[transfer_size:]
            
            self.token_tracker.update(transfer_size)

        # Préparer les tenseurs
        total_length = self.block_size * self.batch_size
        
        # Transférer directement sur GPU
        encoder_input = torch.tensor(
            self.token_buffer[:total_length], 
            dtype=torch.long, 
            device=self.device
        ).view(self.batch_size, self.block_size)
        
        decoder_input = torch.tensor(
            self.token_buffer[total_length:total_length*2], 
            dtype=torch.long, 
            device=self.device
        ).view(self.batch_size, self.block_size)
        
        target = torch.tensor(
            self.token_buffer[total_length+1:total_length*2+1], 
            dtype=torch.long, 
            device=self.device
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