import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import asyncio
from functools import partial
from queue import Queue
import threading
import random
import os
import time

class BatchBuffer:
    """Un gestionnaire de buffer qui précharge des lots complets de données pour gradient accumulation"""
    def __init__(self, capacity=4):
        self.capacity = capacity
        self.buffer = []
        self.lock = threading.Lock()
        self.not_full = threading.Condition(self.lock)
        self.not_empty = threading.Condition(self.lock)
        self.is_closed = False
    
    def put(self, batch_group):
        """Ajoute un groupe de batch (pour une itération d'accumulation complète)"""
        with self.lock:
            while len(self.buffer) >= self.capacity and not self.is_closed:
                self.not_full.wait(timeout=1.0)
                if self.is_closed:
                    return False
            
            if self.is_closed:
                return False
                
            self.buffer.append(batch_group)
            self.not_empty.notify()
            return True
    
    def get(self):
        """Récupère un groupe de batch complet"""
        with self.lock:
            while len(self.buffer) == 0 and not self.is_closed:
                self.not_empty.wait(timeout=1.0)
                if self.is_closed and len(self.buffer) == 0:
                    return None
            
            if len(self.buffer) == 0:
                return None
                
            batch_group = self.buffer.pop(0)
            self.not_full.notify()
            return batch_group
    
    def close(self):
        """Ferme le buffer et réveille tous les threads en attente"""
        with self.lock:
            self.is_closed = True
            self.not_empty.notify_all()
            self.not_full.notify_all()
    
    def __len__(self):
        with self.lock:
            return len(self.buffer)

class FinewebDataset(IterableDataset):
    def __init__(self, split='train', max_length=128, buffer_size=5, shuffle=True, start_offset=0, tokenizer=None, batch_size=4, gradient_accumulation_steps=1):
        super().__init__()
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="CC-MAIN-2024-10",
            split=split,
            streaming=True
        ).skip(start_offset)
        
        # Use provided tokenizer or create new one
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            access_token = os.getenv('HF_TOKEN')
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_fast=True, access_token=access_token)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.start_offset = start_offset
        
        # Nouveau: stocke le nombre d'étapes d'accumulation de gradient
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Utilise un buffer spécial pour les groupes de batch
        self.batch_buffer = BatchBuffer(capacity=max(4, buffer_size))
        
        # Gestionnaire du thread de préchargement
        self.prefetch_thread = None
        self.should_stop = threading.Event()
        
        # Compteurs pour le debugging
        self.batches_prepared = 0
        self.batches_served = 0
        self.current_batch_group = None
        self.current_batch_index = 0
        
        # Démarrer le préchargement immédiatement
        self._start_prefetching()
        
        # Attendre un peu que le premier batch soit chargé
        print("Initializing data loader, waiting for first batch...")
        timeout = 30  # secondes maximum à attendre
        start_time = time.time()
        while len(self.batch_buffer) == 0 and time.time() - start_time < timeout:
            time.sleep(0.1)
        print(f"Data loader initialized, buffer has {len(self.batch_buffer)} batch groups ready")

    def _tokenize_batch(self, texts):
        # Tokenize as encoder-decoder pairs
        tokenized = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # For autoregressive models like MLA:
        # input_ids: the input sequence
        # labels: shifted by one position (next token prediction)
        input_ids = tokenized['input_ids']
        
        # Create labels by shifting input_ids left by 1
        # labels[i] should be input_ids[i+1]
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -1  # Set last position to -1 (will be ignored by loss)
        
        # For consistency with the interface, keep decoder_input_ids same as input_ids
        tokenized['decoder_input_ids'] = input_ids.clone()
        tokenized['decoder_attention_mask'] = tokenized['attention_mask'].clone()
        tokenized['labels'] = labels
        
        # Ensure all tensors are contiguous for better performance
        return {k: v.contiguous() if isinstance(v, torch.Tensor) else v 
                for k, v in tokenized.items()}

    def _prefetch_data(self):
        """Background thread function that continuously prefetches data in batch groups"""
        try:
            dataset_iter = iter(self.dataset)
            
            while not self.should_stop.is_set():
                try:
                    # Prépare un groupe complet de batches pour une itération d'accumulation
                    batch_group = []
                    
                    # Accumule autant de batches que nécessaire pour une étape complète
                    for _ in range(self.gradient_accumulation_steps):
                        buffer = []
                        # Collecte assez d'exemples pour un batch
                        for _ in range(self.batch_size):
                            if self.should_stop.is_set():
                                break
                            
                            example = next(dataset_iter)
                            buffer.append(example['text'])
                        
                        if self.should_stop.is_set():
                            break
                            
                        # Applique le mélange au niveau du batch
                        if self.shuffle:
                            random.shuffle(buffer)
                        
                        # Tokenize ce batch et l'ajoute au groupe
                        tokenized = self._tokenize_batch(buffer)
                        batch_group.append(tokenized)
                        
                        # Compte les batches préparés pour le monitoring
                        self.batches_prepared += 1
                    
                    # Ajoute le groupe complet au buffer
                    if batch_group and not self.should_stop.is_set():
                        if not self.batch_buffer.put(batch_group):
                            break  # Buffer fermé
                    
                except StopIteration:
                    # Dataset épuisé, on met le groupe partiel s'il existe
                    if batch_group:
                        self.batch_buffer.put(batch_group)
                    # Signal de fin
                    self.batch_buffer.close()
                    break
                
                except Exception as e:
                    print(f"Exception in prefetch thread: {e}")
                    # En cas d'erreur, on signale la fin
                    self.batch_buffer.close()
                    break
                    
        except Exception as e:
            print(f"Fatal error in prefetch thread: {e}")
            self.batch_buffer.close()

    def _start_prefetching(self):
        """Start the prefetching thread"""
        if self.prefetch_thread is None or not self.prefetch_thread.is_alive():
            self.should_stop.clear()
            self.prefetch_thread = threading.Thread(
                target=self._prefetch_data, 
                daemon=True,
                name="DataPrefetchThread"
            )
            self.prefetch_thread.start()
            print(f"Started prefetch thread (id: {self.prefetch_thread.ident})")

    def __iter__(self):
        # Make sure prefetching is active
        self._start_prefetching()
        # Réinitialise le traqueur de batch courant
        self.current_batch_group = None
        self.current_batch_index = 0
        return self
    
    def __next__(self):
        # Si nous avons un groupe de batches et qu'il reste des batches dedans, on utilise le suivant
        if self.current_batch_group is not None and self.current_batch_index < len(self.current_batch_group):
            batch = self.current_batch_group[self.current_batch_index]
            self.current_batch_index += 1
            self.batches_served += 1
            return batch
        
        # Sinon, on récupère un nouveau groupe de batches
        self.current_batch_group = self.batch_buffer.get()
        
        # Si None, c'est la fin du dataset
        if self.current_batch_group is None:
            self.should_stop.set()
            raise StopIteration
        
        # Utilise le premier batch du nouveau groupe
        self.current_batch_index = 1
        self.batches_served += 1
        return self.current_batch_group[0]

    def __del__(self):
        # Clean shutdown
        if hasattr(self, 'should_stop'):
            self.should_stop.set()
        
        if hasattr(self, 'batch_buffer'):
            self.batch_buffer.close()
            
        # Wait for thread to finish (with timeout)
        if hasattr(self, 'prefetch_thread') and self.prefetch_thread is not None:
            self.prefetch_thread.join(timeout=0.5)

def async_iterator_to_generator(ait):
    async def wrapper():
        async for item in ait:
            yield item
    return wrapper().__aiter__()
