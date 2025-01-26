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
    def __init__(self, cache_dir='data/token_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.current_cache = None
        self.cache_index = 0
        self.preload_thread = None
        self.next_cache = None
        self.cache_ready = threading.Event()
        self.cache_queue = Queue(maxsize=2)  # Pour stocker le cache actuel et le suivant

    def get_cache_path(self, split, start_iter):
        return os.getenv('HF_TOKEN')

    def save_cache(self, tokens, split, start_iter):
        cache_path = self.get_cache_path(split, start_iter)
        with open(cache_path, 'wb') as f:
            pickle.dump(tokens, f)

    def load_cache(self, split, start_iter):
        cache_path = self.get_cache_path(split, start_iter)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

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
        self.dataset = self.dataset.shuffle(buffer_size=50_000)
        
        # Initialize token tracker
        tracker_dir = os.path.join('data', 'token_tracking')
        os.makedirs(tracker_dir, exist_ok=True)
        self.token_tracker = TokenTracker()
        
        # Modifions la façon dont nous gérons le dataset
        self.dataset_iterator = iter(self.dataset)
        self.token_buffer = []
        
        # Save tokenizer metadata
        self.save_meta()
        
        self.cache_size = cache_size
        self.eval_cache_size = eval_cache_size
        self.token_cache = TokenCache()
        self.current_iter = 0
        self.is_eval_mode = False
        
        # Précharger plus de données en mémoire
        self.prefetch_size = 10  # Nombre de batches à précharger
        self.prefetch_queue = Queue(maxsize=self.prefetch_size)
        self.prefetch_thread = None
        self.start_prefetching()
        
        # Précharger le premier cache au démarrage
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
        """Tokenize a single example."""
        ids = self.tokenizer.encode(example['text'], add_special_tokens=False)
        ids.append(self.tokenizer.eos_token_id)
        return ids
    
    def prepare_initial_cache(self):
        """Prépare le cache initial s'il n'existe pas"""
        cache_path = self.token_cache.get_cache_path(self.split, 0)
        if not os.path.exists(cache_path):
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

    def start_prefetching(self):
        """Démarre le thread de préchargement"""
        def prefetch_worker():
            while True:
                try:
                    batch = self._prepare_batch()
                    if not self.prefetch_queue.full():
                        self.prefetch_queue.put(batch)
                except Exception as e:
                    print(f"Prefetch error: {e}")
                    continue

        self.prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        self.prefetch_thread.start()

    def _prepare_batch(self):
        """Prépare un batch de données"""
        while len(self.token_buffer) < (self.block_size * self.batch_size * 2 + 1):
            example = next(self.dataset_iterator)
            tokens = self.process_example(example)
            self.token_buffer.extend(tokens)
            self.token_tracker.update(len(tokens))

        total_length = self.block_size * self.batch_size
        
        # Préparer les tenseurs sur CPU d'abord
        encoder_input = torch.tensor(self.token_buffer[:total_length], 
                                   dtype=torch.long, device='cpu', pin_memory=True)
        decoder_input = torch.tensor(self.token_buffer[total_length:total_length*2], 
                                   dtype=torch.long, device='cpu', pin_memory=True)
        target = torch.tensor(self.token_buffer[total_length+1:total_length*2+1], 
                            dtype=torch.long, device='cpu', pin_memory=True)

        # Reshape
        encoder_input = encoder_input.view(self.batch_size, self.block_size)
        decoder_input = decoder_input.view(self.batch_size, self.block_size)
        target = target.view(self.batch_size, self.block_size)

        # Nettoie le buffer
        self.token_buffer = self.token_buffer[total_length*2:]
        
        return encoder_input, decoder_input, target

    def get_batch(self):
        """Get a batch of token sequences of length block_size."""
        # Récupérer un batch préchargé
        encoder_input, decoder_input, target = self.prefetch_queue.get()
        
        # Transférer sur GPU de manière asynchrone
        with torch.cuda.stream(torch.cuda.Stream()):
            encoder_input = encoder_input.cuda(non_blocking=True)
            decoder_input = decoder_input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

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