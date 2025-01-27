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
        self.dataset = self.dataset.shuffle(buffer_size=10_000)
        
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

    def get_batch(self):
        """Get a batch of token sequences of length block_size."""
        cache_size = self.eval_cache_size if self.is_eval_mode else self.cache_size
        
        # Si on arrive à la fin du cache actuel
        if self.current_iter % cache_size == 0:
            # Attendre que le prochain cache soit prêt
            self.token_cache.cache_ready.wait()
            self.token_cache.cache_ready.clear()
            
            # Récupérer le cache
            self.current_cache = self.token_cache.cache_queue.get()
            
            # Précharger le prochain cache si on n'est pas en mode eval
            if not self.is_eval_mode:
                next_start_iter = self.current_iter + cache_size
                self.token_cache.preload_next_cache(self.split, next_start_iter)
        
        # Utiliser les tokens du cache
        cache_index = self.current_iter % cache_size
        tokens = self.current_cache[cache_index]
        self.token_buffer.extend(tokens)
        self.token_tracker.update(len(tokens))
        
        # On s'assure d'avoir assez de tokens dans le buffer
        # tracking time to get a batch
        start_time = time.time()
        while len(self.token_buffer) < (self.block_size * self.batch_size * 2 + 1):
            try:
                example = next(self.dataset_iterator)
                new_tokens = self.process_example(example)
                self.token_buffer.extend(new_tokens)
                self.token_tracker.update(len(new_tokens))
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
                    continue
                # Sinon, on peut utiliser ce qu'il reste dans le buffer
                break

        # Si après avoir essayé de remplir le buffer, on n'a toujours pas assez de tokens,
        # on utilise ce qu'on a en ajustant la taille du batch
        available_sequences = len(self.token_buffer) // self.block_size
        if available_sequences == 0:
            raise RuntimeError("Not enough tokens to create even one sequence")
        
        actual_batch_size = min(self.batch_size, available_sequences // 2)  # Divise par 2 car on a besoin d'input et target
        total_length = self.block_size * actual_batch_size

        # Prepare les tensors comme avant, mais avec la taille de batch ajustée
        encoder_input = torch.tensor(self.token_buffer[:total_length], 
                                   dtype=torch.long, device=self.device)
        decoder_input = torch.tensor(self.token_buffer[total_length:total_length*2], 
                                   dtype=torch.long, device=self.device)
        target = torch.tensor(self.token_buffer[total_length+1:total_length*2+1], 
                            dtype=torch.long, device=self.device)

        # Reshape avec la taille de batch ajustée
        encoder_input = encoder_input.view(actual_batch_size, self.block_size)
        decoder_input = decoder_input.view(actual_batch_size, self.block_size)
        target = target.view(actual_batch_size, self.block_size)

        # Nettoie le buffer
        self.token_buffer = self.token_buffer[total_length*2:]
        
        end_time = time.time()
        # print(f"Time to get a batch: {end_time - start_time:.4f} seconds")

        self.current_iter += 1
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