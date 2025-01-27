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
        
        # Prefetch queue size
        self.prefetch_factor = 3
        self.prefetch_queue = Queue(maxsize=self.prefetch_factor)
        self.shutdown_event = threading.Event()
        
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
        tracker_dir = os.path.join('data', 'token_tracking')
        os.makedirs(tracker_dir, exist_ok=True)
        self.token_tracker = TokenTracker()
        
        self.dataset_iterator = iter(self.dataset)
        self.token_buffer = []
        
        # Save tokenizer metadata
        self.save_meta()
        
        self.cache_size = cache_size
        self.eval_cache_size = eval_cache_size
        self.token_cache = TokenCache()
        self.current_iter = 0
        self.is_eval_mode = False
        
        # Start prefetch thread
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
        
        # Prepare initial cache
        if split == 'train':
            self.prepare_initial_cache()
            
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
    
    def prepare_initial_cache(self):
        """Prépare le cache initial s'il n'existe pas"""
        if self.token_cache.load_cache(self.split, 0) is None:
            print(f"Preparing initial cache for {self.split}...")
            tokens = []
            for _ in range(self.cache_size):
                example = next(self.dataset_iterator)
                tokens.append(self.process_example(example))
            self.token_cache.save_cache(tokens, self.split, 0)
            print("Initial cache prepared")
        
        # Précharger le premier cache
        self.token_cache.preload_next_cache(self.split, 0)

    def set_eval_mode(self, is_eval):
        """Bascule entre mode évaluation et entraînement"""
        self.is_eval_mode = is_eval
        if is_eval:
            # Charger le cache d'évaluation si nécessaire
            eval_cache = self.token_cache.load_cache(f"{self.split}_eval", 0)
            if eval_cache is None:
                print("Preparing eval cache...")
                eval_tokens = []
                for _ in range(self.eval_cache_size):
                    example = next(self.dataset_iterator)
                    eval_tokens.append(self.process_example(example))
                self.token_cache.save_cache(eval_tokens, f"{self.split}_eval", 0)
                print("Eval cache prepared")

    def _prefetch_worker(self):
        """Worker thread to prefetch and prepare batches"""
        while not self.shutdown_event.is_set():
            try:
                if self.prefetch_queue.qsize() < self.prefetch_factor:
                    # Prepare next batch
                    batch = self._prepare_batch()
                    if batch is not None:
                        self.prefetch_queue.put(batch)
            except Exception as e:
                print(f"Prefetch worker error: {e}")
                continue
            time.sleep(0.001)  # Small sleep to prevent busy waiting
            
    def _prepare_batch(self):
        """Prepare a single batch in the background"""
        try:
            # Get data from cache or dataset
            if self.current_iter % self.cache_size == 0:
                self.token_cache.cache_ready.wait()
                self.current_cache = self.token_cache.cache_queue.get()
                if not self.is_eval_mode:
                    next_start_iter = self.current_iter + self.cache_size
                    self.token_cache.preload_next_cache(self.split, next_start_iter)
            
            # Process batch
            cache_index = self.current_iter % self.cache_size
            tokens = self.current_cache[cache_index]
            
            # Prepare tensors on CPU first
            encoder_input = torch.tensor(tokens[:self.block_size], dtype=torch.long)
            decoder_input = torch.tensor(tokens[self.block_size:self.block_size*2], dtype=torch.long)
            target = torch.tensor(tokens[self.block_size+1:self.block_size*2+1], dtype=torch.long)
            
            # Pin memory for faster transfer
            if self.device != 'cpu':
                encoder_input = encoder_input.pin_memory()
                decoder_input = decoder_input.pin_memory()
                target = target.pin_memory()
            
            return encoder_input, decoder_input, target
            
        except Exception as e:
            print(f"Batch preparation error: {e}")
            return None
            
    def get_batch(self):
        """Get a batch with prefetching"""
        try:
            # Get pre-fetched batch
            encoder_input, decoder_input, target = self.prefetch_queue.get()
            
            # Transfer to device in parallel with next batch preparation
            if self.device != 'cpu':
                encoder_input = encoder_input.to(self.device, non_blocking=True)
                decoder_input = decoder_input.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
            
            self.current_iter += 1
            return encoder_input, decoder_input, target
            
        except Exception as e:
            print(f"Error getting batch: {e}")
            # Fallback to synchronous batch preparation
            return self._prepare_batch()
            
    def __iter__(self):
        while True:
            yield self.get_batch()
    
    def get_total_tokens(self):
        """Return the total number of tokens processed so far."""
        return self.token_tracker.total_tokens

    def __del__(self):
        self.shutdown_event.set()
        if hasattr(self, 'prefetch_thread'):
            self.prefetch_thread.join(timeout=1.0)

    def worker_init_fn(worker_id):
        """Initialize worker with unique seed"""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Set different seed for each worker
            torch.manual_seed(worker_info.seed % 2**32)
            np.random.seed(worker_info.seed % 2**32)
            
            # Each worker should get its own iterator
            dataset = worker_info.dataset
            dataset.dataset_iterator = iter(dataset.dataset)
            
    @staticmethod
    def get_dataloader(dataset, num_workers=4, pin_memory=True, persistent_workers=True):
        """Create an optimized DataLoader for the streaming dataset"""
        # Create a context for the dataloader that ensures proper CUDA behavior
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