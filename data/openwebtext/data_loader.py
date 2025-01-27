import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset
import numpy as np
import pickle
import time
import threading
from queue import Queue

class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
    
    def update(self, new_tokens):
        self.total_tokens += new_tokens

class TokenCache:
    def __init__(self):
        self.cache_dict = {}
        self.preload_thread = None
        self.cache_ready = threading.Event()
        self.cache_queue = Queue(maxsize=2)

    def save_cache(self, tokens, split, start_iter):
        cache_key = f"{split}_{start_iter}"
        self.cache_dict[cache_key] = tokens

    def load_cache(self, split, start_iter):
        cache_key = f"{split}_{start_iter}"
        return self.cache_dict.get(cache_key)

    def preload_next_cache(self, split, start_iter):
        """Précharge le prochain cache dans un thread séparé"""
        def _preload():
            next_cache = self.load_cache(split, start_iter)
            if next_cache is not None:
                self.cache_queue.put(next_cache)
                self.cache_ready.set()

        self.preload_thread = threading.Thread(target=_preload)
        self.preload_thread.daemon = True
        self.preload_thread.start()

class StreamingDataset(IterableDataset):
    def __init__(self, block_size, batch_size, dataset_name="HuggingFaceFW/fineweb-edu", 
                 dataset_config="CC-MAIN-2024-10", split="train", device='cuda',
                 cache_size=1000, eval_cache_size=100):
        super().__init__()
        
        # Set multiprocessing method to spawn for CUDA compatibility
        if device == 'cuda' and torch.multiprocessing.get_start_method() != 'spawn':
            try:
                torch.multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError:
                pass
                
        # Set tokenizer parallelism explicitly
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.dataset_config = dataset_config
        self.split = split
        
        access_token = os.getenv('HF_TOKEN')
        
        # Initialize tokenizer with proper max length
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct", 
            use_fast=True,
            access_token=access_token,
            model_max_length=block_size,
            padding_side='right',
            truncation=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize dataset without num_proc for streaming
        self.dataset = load_dataset(
            dataset_name, 
            name=dataset_config, 
            split=split, 
            streaming=True
        )
        self.dataset = self.dataset.shuffle(buffer_size=20_000)
        
        # Initialize token tracker
        self.total_tokens = 0
        
        # Worker-specific state
        self.worker_id = None
        self.num_workers = None
        self.worker_seed = None
        
        # Buffer for tokens
        self.token_buffer = []
        
    def process_example(self, example):
        """Tokenize a single example with proper truncation."""
        # Tokenize with truncation
        tokens = self.tokenizer.encode(
            example['text'],
            add_special_tokens=False,
            truncation=True,
            max_length=self.block_size * 2  # Double block size to account for both encoder and decoder
        )
        
        # Add EOS token
        tokens.append(self.tokenizer.eos_token_id)
        
        # Ensure we have enough tokens
        if len(tokens) < self.block_size * 2:
            # Pad with EOS tokens if needed
            tokens.extend([self.tokenizer.eos_token_id] * (self.block_size * 2 - len(tokens)))
        
        return tokens[:self.block_size * 2]  # Ensure exact length
    
    def _init_worker(self):
        """Initialize worker-specific state"""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
            self.worker_seed = worker_info.seed
            
            # Set worker-specific random seeds
            torch.manual_seed(self.worker_seed)
            np.random.seed(self.worker_seed)
            
            # Get worker-specific iterator
            self.dataset_iterator = iter(self.dataset)
            
            # Skip examples to distribute among workers
            for _ in range(self.worker_id):
                next(self.dataset_iterator)
    
    def __iter__(self):
        # Initialize worker if not done yet
        if self.worker_id is None:
            self._init_worker()
        
        while True:
            # Fill buffer if needed
            while len(self.token_buffer) < self.block_size * 2:
                try:
                    example = next(self.dataset_iterator)
                    tokens = self.process_example(example)
                    self.token_buffer.extend(tokens)
                    self.total_tokens += len(tokens)
                    
                    # Skip examples for other workers
                    if self.num_workers is not None:
                        for _ in range(self.num_workers - 1):
                            next(self.dataset_iterator)
                except StopIteration:
                    # Reset iterator if we reach the end
                    self.dataset_iterator = iter(self.dataset)
                    if len(self.token_buffer) == 0:
                        continue
                    break
            
            # Prepare batch
            if len(self.token_buffer) >= self.block_size * 2:
                # Extract sequences
                encoder_input = torch.tensor(
                    self.token_buffer[:self.block_size], 
                    dtype=torch.long
                )
                decoder_input = torch.tensor(
                    self.token_buffer[self.block_size:self.block_size*2], 
                    dtype=torch.long
                )
                target = torch.tensor(
                    self.token_buffer[self.block_size+1:self.block_size*2+1], 
                    dtype=torch.long
                )
                
                # Update buffer
                self.token_buffer = self.token_buffer[self.block_size*2:]
                
                # Pin memory if using CUDA
                if self.device != 'cpu':
                    encoder_input = encoder_input.pin_memory()
                    decoder_input = decoder_input.pin_memory()
                    target = target.pin_memory()
                
                yield encoder_input, decoder_input, target
            else:
                # Not enough tokens left, start over
                self.token_buffer = []
                continue
    
    def get_total_tokens(self):
        return self.total_tokens

    @staticmethod
    def get_dataloader(dataset, num_workers=4, pin_memory=True, persistent_workers=True):
        """Create an optimized DataLoader for the streaming dataset"""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=None,  # Dataset already handles batching
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            multiprocessing_context='spawn'  # Force spawn for CUDA compatibility
        )

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