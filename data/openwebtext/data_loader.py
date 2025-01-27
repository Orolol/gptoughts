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
        self.dataset_name = dataset_name
        
        # Initialize these in worker_init
        self.tokenizer = None
        self.dataset = None
        self.dataset_iterator = None
        self.token_buffer = []
        self.current_iter = 0
        self.is_eval_mode = False
        
        # Save configuration for workers
        self.cache_size = cache_size
        self.eval_cache_size = eval_cache_size
        
    def _init_worker(self):
        """Initialize worker-specific resources"""
        if self.tokenizer is None:
            access_token = os.getenv('HF_TOKEN')
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-1B-Instruct", 
                use_fast=True,
                access_token=access_token,
                model_max_length=self.block_size,
                padding_side='right',
                truncation=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        if self.dataset is None:
            self.dataset = load_dataset(
                self.dataset_name,
                name=self.dataset_config,
                split=self.split,
                streaming=True
            )
            self.dataset = self.dataset.shuffle(buffer_size=20_000)
            self.dataset_iterator = iter(self.dataset)
            
    def process_example(self, example):
        """Tokenize a single example with proper truncation."""
        # Ensure worker is initialized
        self._init_worker()
        
        # Tokenize with truncation
        tokens = self.tokenizer.encode(
            example['text'],
            add_special_tokens=False,
            truncation=True,
            max_length=self.block_size * 2
        )
        
        # Add EOS token
        tokens.append(self.tokenizer.eos_token_id)
        
        # Ensure we have enough tokens
        if len(tokens) < self.block_size * 2:
            tokens.extend([self.tokenizer.eos_token_id] * (self.block_size * 2 - len(tokens)))
        
        return tokens[:self.block_size * 2]
    
    def get_batch(self):
        """Get a single batch of data"""
        # Ensure worker is initialized
        self._init_worker()
        
        try:
            # Get next example
            example = next(self.dataset_iterator)
            tokens = self.process_example(example)
            
            # Create tensors
            encoder_input = torch.tensor(tokens[:self.block_size], dtype=torch.long)
            decoder_input = torch.tensor(tokens[self.block_size:self.block_size*2], dtype=torch.long)
            target = torch.tensor(tokens[self.block_size+1:self.block_size*2+1], dtype=torch.long)
            
            # Pin memory if using CUDA
            if self.device != 'cpu':
                encoder_input = encoder_input.pin_memory()
                decoder_input = decoder_input.pin_memory()
                target = target.pin_memory()
            
            return encoder_input, decoder_input, target
            
        except StopIteration:
            # Reset iterator
            self.dataset_iterator = iter(self.dataset)
            return self.get_batch()
    
    def __iter__(self):
        # Ensure worker is initialized
        self._init_worker()
        return self
    
    def __next__(self):
        return self.get_batch()
    
    @staticmethod
    def worker_init_fn(worker_id):
        """Initialize worker with unique seed"""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Set different seed for each worker
            torch.manual_seed(worker_info.seed % 2**32)
            np.random.seed(worker_info.seed % 2**32)
            
            # Initialize worker's dataset
            dataset = worker_info.dataset
            dataset._init_worker()
    
    @staticmethod
    def get_dataloader(dataset, num_workers=4, pin_memory=True, persistent_workers=True):
        """Create an optimized DataLoader for the streaming dataset"""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=None,  # Dataset already handles batching
            num_workers=num_workers,
            worker_init_fn=StreamingDataset.worker_init_fn,
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