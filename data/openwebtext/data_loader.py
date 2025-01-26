import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset
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
        
        access_token = os.getenv('HF_TOKEN')
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_fast=True, access_token=access_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize dataset
        self.dataset = load_dataset(dataset_name, name=dataset_config, split=split, streaming=True)
        self.dataset = self.dataset.shuffle(buffer_size=10_000)
        
        # Initialize token tracker
        tracker_dir = os.path.join('data', 'token_tracking')
        os.makedirs(tracker_dir, exist_ok=True)
        self.token_tracker = TokenTracker()
        
        # Modifions la façon dont nous gérons le dataset
        self.dataset_iterator = iter(self.dataset)
        
        # Ajouter un buffer préfetch plus grand
        self.prefetch_buffer_size = 20_000  # Augmenté pour réduire les pauses
        self.token_buffer = []
        
        # Créer un stream CUDA dédié pour le préfetch
        self.prefetch_stream = torch.cuda.Stream()
        
        # Allouer des buffers fixes pour les tenseurs
        # D'abord créer sur CPU avec pin_memory, puis transférer sur GPU
        self.encoder_buffer_cpu = torch.empty(
            (batch_size, block_size), 
            dtype=torch.long,
            pin_memory=True
        )
        self.decoder_buffer_cpu = torch.empty_like(self.encoder_buffer_cpu, pin_memory=True)
        self.target_buffer_cpu = torch.empty_like(self.encoder_buffer_cpu, pin_memory=True)
        
        # Créer les buffers GPU
        self.encoder_buffer = torch.empty(
            (batch_size, block_size), 
            dtype=torch.long,
            device=device
        )
        self.decoder_buffer = torch.empty_like(self.encoder_buffer)
        self.target_buffer = torch.empty_like(self.encoder_buffer)
        
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
        with torch.cuda.stream(self.prefetch_stream):
            try:
                # Remplir le buffer en arrière-plan
                while len(self.token_buffer) < self.prefetch_buffer_size:
                    example = next(self.dataset_iterator)
                    new_tokens = self.process_example(example)
                    self.token_buffer.extend(new_tokens)
                    self.token_tracker.update(len(new_tokens))
                
                # Préparer les données dans les buffers CPU pré-alloués
                total_length = self.block_size * self.batch_size
                
                # Copier d'abord dans les buffers CPU
                self.encoder_buffer_cpu.copy_(
                    torch.tensor(
                        self.token_buffer[:total_length], 
                        dtype=torch.long
                    ).view(self.batch_size, self.block_size)
                )
                self.decoder_buffer_cpu.copy_(
                    torch.tensor(
                        self.token_buffer[total_length:total_length*2], 
                        dtype=torch.long
                    ).view(self.batch_size, self.block_size)
                )
                self.target_buffer_cpu.copy_(
                    torch.tensor(
                        self.token_buffer[total_length+1:total_length*2+1], 
                        dtype=torch.long
                    ).view(self.batch_size, self.block_size)
                )
                
                # Transférer de manière asynchrone vers GPU
                self.encoder_buffer.copy_(self.encoder_buffer_cpu, non_blocking=True)
                self.decoder_buffer.copy_(self.decoder_buffer_cpu, non_blocking=True)
                self.target_buffer.copy_(self.target_buffer_cpu, non_blocking=True)
                
                # Nettoyer le buffer
                self.token_buffer = self.token_buffer[total_length*2:]
                
                # Synchroniser le stream de préfetch
                self.prefetch_stream.synchronize()
                
                return (
                    self.encoder_buffer.clone(), 
                    self.decoder_buffer.clone(), 
                    self.target_buffer.clone()
                )
                
            except StopIteration:
                # Quand on arrive à la fin du dataset, on crée un nouvel itérateur
                self.dataset = load_dataset("HuggingFaceFW/fineweb-2", 
                                         num_proc=4,
                                         name=self.dataset_config,
                                         split=self.split,
                                         streaming=True)
                self.dataset = self.dataset.shuffle(buffer_size=10_000)
                self.dataset_iterator = iter(self.dataset)
                # Si le buffer est vide après avoir atteint la fin du dataset,
                # on continue à charger des données
                if len(self.token_buffer) == 0:
                    raise StopIteration("End of dataset reached")
                # Sinon, on peut utiliser ce qu'il reste dans le buffer
                return self.get_batch()
    
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