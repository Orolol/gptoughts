import os
import torch
from torch.utils.data import IterableDataset
import time
import threading
import queue
from contextlib import nullcontext
from typing import List, Dict, Optional, Union, Tuple
import random
import gc

class TokenBufferManager:
    """Manages a buffer of token data for efficient pre-fetching"""
    
    def __init__(self, buffer_size: int = 64, device: str = 'cuda'):
        self.buffer_size = buffer_size
        self.device = device
        self.buffer = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.prefetch_thread = None
        self.prefetch_queue = queue.Queue(maxsize=buffer_size)
        self.current_position = 0
    
    def start_prefetch(self, data_source, batch_size, max_length):
        """Start prefetching thread for asynchronous data loading"""
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            return  # Already running
            
        def prefetch_worker():
            try:
                while not self.stop_event.is_set():
                    try:
                        # Get a batch of data
                        batch = next(data_source)
                        
                        # Process the batch if necessary
                        if isinstance(batch, dict):
                            # Put the batch in the queue
                            self.prefetch_queue.put(batch, block=True, timeout=5.0)
                        else:
                            # Create a batch dictionary
                            processed_batch = {"input_ids": batch, "labels": batch}
                            self.prefetch_queue.put(processed_batch, block=True, timeout=5.0)
                    except StopIteration:
                        # End of data source
                        break
                    except queue.Full:
                        # Queue is full, continue trying
                        continue
                    except Exception as e:
                        print(f"Prefetch error: {e}")
                        break
            finally:
                # Signal end of data
                self.stop_event.set()
        
        # Start prefetch thread
        self.stop_event.clear()
        self.prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        self.prefetch_thread.start()
    
    def get_batch(self, timeout: float = 5.0) -> Dict[str, torch.Tensor]:
        """Get a batch from the buffer"""
        try:
            batch = self.prefetch_queue.get(block=True, timeout=timeout)
            return batch
        except queue.Empty:
            if self.stop_event.is_set() and self.prefetch_queue.empty():
                raise StopIteration("No more data in buffer")
            raise queue.Empty("Buffer empty, but prefetch still running")
    
    def get_accumulation_batches(self, num_batches: int) -> List[Dict[str, torch.Tensor]]:
        """Get multiple batches for gradient accumulation"""
        batches = []
        for _ in range(num_batches):
            try:
                batch = self.get_batch()
                batches.append(batch)
            except (StopIteration, queue.Empty):
                # If we've gotten at least one batch, return what we have
                if batches:
                    break
                else:
                    raise StopIteration("No more data available")
        return batches
    
    def stop(self):
        """Stop the prefetch thread"""
        self.stop_event.set()
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=1.0)
        # Clear the queue
        while not self.prefetch_queue.empty():
            try:
                self.prefetch_queue.get_nowait()
            except:
                pass

class LLMDataLoader:
    """Efficient data loader for LLaDA model training"""
    
    def __init__(self, tokenizer, batch_size: int = 32, max_length: int = 1024,
                 buffer_size: int = 64, device: str = 'cuda', 
                 gradient_accumulation_steps: int = 1, num_workers: int = 2):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_workers = num_workers
        
        # Create buffer manager
        self.buffer_manager = TokenBufferManager(buffer_size=buffer_size, device=device)
        
        # Setup data source
        self.setup_data_source()
    
    def setup_data_source(self):
        """Setup the data source - override in subclasses"""
        # This should be implemented by subclasses
        pass
    
    def __iter__(self):
        # Start prefetching
        self.buffer_manager.start_prefetch(
            self.data_source,
            self.batch_size,
            self.max_length
        )
        return self
    
    def __next__(self) -> Dict[str, torch.Tensor]:
        """Get next batch from the buffer"""
        try:
            return self.buffer_manager.get_batch()
        except StopIteration:
            # Restart data source and buffer
            self.buffer_manager.stop()
            self.setup_data_source()
            self.buffer_manager.start_prefetch(
                self.data_source,
                self.batch_size,
                self.max_length
            )
            # Try again
            return self.buffer_manager.get_batch()
    
    def get_accumulation_batches(self, num_batches: int) -> List[Dict[str, torch.Tensor]]:
        """Get multiple batches for gradient accumulation"""
        return self.buffer_manager.get_accumulation_batches(num_batches)
    
    def __del__(self):
        # Clean up resources
        if hasattr(self, 'buffer_manager'):
            self.buffer_manager.stop()

class HuggingFaceDataLoader(LLMDataLoader):
    """Data loader for Hugging Face datasets"""
    
    def __init__(self, tokenizer, batch_size: int = 32, max_length: int = 1024,
                 buffer_size: int = 64, device: str = 'cuda',
                 gradient_accumulation_steps: int = 1, num_workers: int = 2,
                 dataset_name: str = "HuggingFaceFW/fineweb-edu", dataset_split: str = "train"):
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        
        super().__init__(
            tokenizer, batch_size, max_length, buffer_size, 
            device, gradient_accumulation_steps, num_workers
        )
    
    def setup_data_source(self):
        """Setup the Hugging Face dataset"""
        try:
            from datasets import load_dataset
            
            # Load dataset
            dataset = load_dataset(self.dataset_name, split=self.dataset_split, streaming=True)
            
            def data_generator():
                for example in dataset:
                    if 'text' in example:
                        text = example['text']
                        # Tokenize
                        tokens = self.tokenizer(
                            text, 
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors="pt"
                        )
                        
                        # Prepare labels
                        input_ids = tokens['input_ids'][0]
                        labels = input_ids.clone()
                        
                        # Create batch
                        batch = {
                            "input_ids": input_ids,
                            "labels": labels
                        }
                        
                        yield batch
            
            self.data_source = data_generator()
        except ImportError:
            raise ImportError("Please install the datasets library: pip install datasets")
        except Exception as e:
            raise RuntimeError(f"Error setting up Hugging Face dataset: {e}")

def create_llm_data_loader(tokenizer, batch_size: int = 32, max_length: int = 1024,
                          gradient_accumulation_steps: int = 1, buffer_size: int = 64,
                          num_workers: int = 2, report_interval: int = 100,
                          dataset_name: str = "HuggingFaceFW/fineweb-edu", 
                          dataset_split: str = "train"):
    """Create an optimized data loader for LLM training"""
    
    # Use Hugging Face dataset loader
    return HuggingFaceDataLoader(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        buffer_size=buffer_size,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_workers=num_workers,
        dataset_name=dataset_name,
        dataset_split=dataset_split
    )
