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
                 dataset_config="CC-MAIN-2024-10", split="train", device='cuda',
                 num_workers=4):
        super().__init__()
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.dataset_config = dataset_config
        self.split = split
        self.num_workers = num_workers
        
        access_token = os.getenv('HF_TOKEN')
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct", 
            use_fast=True, 
            access_token=access_token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize token tracker
        tracker_dir = os.path.join('data', 'token_tracking')
        os.makedirs(tracker_dir, exist_ok=True)
        self.token_tracker = TokenTracker()
        
        # Calculer la taille minimale du buffer de préfetch
        min_tokens_needed = block_size * batch_size * 2  # Pour encoder et decoder
        self.prefetch_buffer_size = max(min_tokens_needed * 2, 50_000)  # Au moins 2x la taille nécessaire
        print(f"Prefetch buffer size: {self.prefetch_buffer_size}")
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
        
        # Initialize dataset and iterator
        self.dataset = None
        self.dataset_iterator = None
        self.reset_dataset()  # This will set up dataset and iterator
    
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
    
    def reset_dataset(self):
        """Reset the dataset iterator"""
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu", 
            name=self.dataset_config,
            split=self.split,
            streaming=True
        )
        self.dataset = self.dataset.shuffle(buffer_size=10_000)
        self.dataset_iterator = iter(self.dataset)
        self.token_buffer = []
    
    def get_batch(self):
        """Get a batch of token sequences of length block_size."""
        max_retries = 3
        retry_count = 0
        required_tokens = self.block_size * self.batch_size * 2
        
        while retry_count < max_retries:
            try:
                with torch.cuda.stream(self.prefetch_stream):
                    # Remplir le buffer en arrière-plan
                    fill_start_time = time.time()
                    
                    fill_attempts = 0
                    max_fill_attempts = 1000
                    
                    # Remplissage rapide sans logs fréquents
                    while len(self.token_buffer) < required_tokens and fill_attempts < max_fill_attempts:
                        try:
                            example = next(self.dataset_iterator)
                            if not example or 'text' not in example:
                                continue
                            
                            new_tokens = self.process_example(example)
                            if not new_tokens:
                                continue
                            
                            self.token_buffer.extend(new_tokens)
                            self.token_tracker.update(len(new_tokens))
                            
                            # Log moins fréquent
                            if fill_attempts % 50 == 0 and fill_attempts > 0:
                                tokens_per_sec = len(self.token_buffer) / (time.time() - fill_start_time)
                                print(f"Tokenization speed: {tokens_per_sec:.0f} tokens/s")
                            
                        except StopIteration:
                            self.reset_dataset()
                        
                        fill_attempts += 1
                    
                    if fill_attempts >= max_fill_attempts or len(self.token_buffer) < required_tokens:
                        retry_count += 1
                        continue
                    
                    # Optimisation des copies mémoire
                    total_length = self.block_size * self.batch_size
                    
                    # Créer les tenseurs une seule fois et les copier directement
                    tokens_tensor = torch.tensor(
                        self.token_buffer[:total_length*2], 
                        dtype=torch.long
                    ).view(-1, self.block_size)
                    
                    # Copier en une seule opération vers CPU
                    with torch.cuda.stream(self.prefetch_stream):
                        self.encoder_buffer_cpu.copy_(tokens_tensor[:self.batch_size])
                        self.decoder_buffer_cpu.copy_(tokens_tensor[self.batch_size:2*self.batch_size])
                        self.target_buffer_cpu.copy_(tokens_tensor[self.batch_size+1:2*self.batch_size+1])
                    
                        # Transfert asynchrone vers GPU
                        self.encoder_buffer.copy_(self.encoder_buffer_cpu, non_blocking=True)
                        self.decoder_buffer.copy_(self.decoder_buffer_cpu, non_blocking=True)
                        self.target_buffer.copy_(self.target_buffer_cpu, non_blocking=True)
                    
                    # Nettoyer le buffer
                    self.token_buffer = self.token_buffer[total_length*2:]
                    
                    # Synchronisation minimale
                    self.prefetch_stream.synchronize()
                    
                    # Log de performance occasionnel (1 fois sur 100)
                    if self.token_tracker.total_tokens % 100000 == 0:
                        current_time = time.time()
                        elapsed = current_time - fill_start_time
                        print(f"Average throughput: {len(self.token_buffer)/elapsed:.0f} tokens/s")
                    
                    return (
                        self.encoder_buffer, 
                        self.decoder_buffer, 
                        self.target_buffer
                    )
                    
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    self.reset_dataset()
                    retry_count = 0
                continue
        
        raise RuntimeError("Failed to get batch after multiple retries")
    
    def __iter__(self):
        while True:
            try:
                yield self.get_batch()
            except Exception as e:
                print(f"Error in iterator: {str(e)}")
                self.reset_dataset()
                continue
    
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