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
        
        # Use a larger queue size and multiple queues for better prefetching
        self.prefetch_queue = queue.Queue(maxsize=buffer_size * 2)
        self.ready_queue = queue.Queue(maxsize=buffer_size)
        self.current_position = 0
        
        # Stats for monitoring
        self.stats = {
            "fetched": 0,
            "processed": 0,
            "queue_wait_time": 0,
            "prefetch_time": 0
        }
        
        # Create dedicated CUDA stream for data transfer
        self.cuda_stream = torch.cuda.Stream() if torch.cuda.is_available() and device == 'cuda' else None
    
    def start_prefetch(self, data_source, batch_size, max_length):
        """Start prefetching thread for asynchronous data loading"""
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            return  # Already running
        
        # Reset stop event
        self.stop_event.clear()
        
        # Create multiple worker threads for better parallelism
        num_workers = min(4, max(1, os.cpu_count() or 4))
        self.workers = []
        
        # Create a shared queue for workers to output to
        raw_queue = queue.Queue(maxsize=self.buffer_size * 2)
        
        # Define worker function for data fetching
        def fetch_worker(worker_id):
            try:
                while not self.stop_event.is_set():
                    try:
                        # Get a batch of data
                        t_start = time.time()
                        batch = next(data_source)
                        t_fetch = time.time() - t_start
                        
                        with self.lock:
                            self.stats["prefetch_time"] += t_fetch
                            self.stats["fetched"] += 1
                        
                        # Process the batch if necessary
                        if isinstance(batch, dict):
                            raw_queue.put(batch, block=True, timeout=5.0)
                        else:
                            # Create a batch dictionary
                            processed_batch = {"input_ids": batch, "labels": batch}
                            raw_queue.put(processed_batch, block=True, timeout=5.0)
                    except StopIteration:
                        # End of data source
                        raw_queue.put(None)  # Signal end of data
                        break
                    except queue.Full:
                        # Queue is full, continue trying
                        continue
                    except Exception as e:
                        print(f"Worker {worker_id} fetch error: {e}")
                        if not self.stop_event.is_set():
                            # Only print stack trace if not shutting down
                            import traceback
                            traceback.print_exc()
                        break
            finally:
                # Signal that this worker is done
                try:
                    raw_queue.put(None, block=False)
                except queue.Full:
                    pass
        
        # Define worker for GPU transfer
        def gpu_transfer_worker():
            try:
                done_count = 0
                while not self.stop_event.is_set() and done_count < num_workers:
                    try:
                        batch = raw_queue.get(block=True, timeout=5.0)
                        if batch is None:
                            # One worker is done
                            done_count += 1
                            continue
                        
                        # Transfer to GPU if needed
                        if self.device == 'cuda' and self.cuda_stream is not None and torch.cuda.is_available():
                            with torch.cuda.stream(self.cuda_stream):
                                # Transfer tensors to GPU
                                for key, tensor in batch.items():
                                    if isinstance(tensor, torch.Tensor) and tensor.device.type != 'cuda':
                                        batch[key] = tensor.to(self.device, non_blocking=True)
                                
                                # Synchronize stream to ensure tensors are transferred
                                self.cuda_stream.synchronize()
                        
                        # Put batch in the ready queue
                        t_start = time.time()
                        self.prefetch_queue.put(batch, block=True, timeout=5.0)
                        
                        with self.lock:
                            self.stats["queue_wait_time"] += time.time() - t_start
                        
                    except queue.Empty:
                        # Timeout, check if we should continue
                        continue
                    except queue.Full:
                        # Queue is full, wait and try again
                        time.sleep(0.1)
                        continue
                    except Exception as e:
                        print(f"GPU transfer error: {e}")
                        if not self.stop_event.is_set():
                            import traceback
                            traceback.print_exc()
            finally:
                # Signal end of data when all workers are done
                self.stop_event.set()
        
        # Start prefetch worker threads
        self.stop_event.clear()
        
        # Start worker threads
        for i in range(num_workers):
            worker = threading.Thread(
                target=fetch_worker,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            
        # Start GPU transfer thread
        gpu_thread = threading.Thread(
            target=gpu_transfer_worker,
            daemon=True
        )
        gpu_thread.start()
        self.prefetch_thread = gpu_thread
    
    def get_batch(self, timeout: float = 5.0) -> Dict[str, torch.Tensor]:
        """Get a batch from the buffer with high performance"""
        try:
            # First check if we have something in ready queue
            if not self.ready_queue.empty():
                return self.ready_queue.get_nowait()
            
            # Otherwise get from prefetch queue with timeout
            t_start = time.time()
            batch = self.prefetch_queue.get(block=True, timeout=timeout)
            
            # Update stats
            with self.lock:
                self.stats["queue_wait_time"] += time.time() - t_start
                self.stats["processed"] += 1
            
            # Check if we should fill ready queue
            if self.ready_queue.qsize() < self.buffer_size // 4:
                # Try to prefill ready queue with a few batches
                try:
                    for _ in range(min(4, self.buffer_size // 4)):
                        next_batch = self.prefetch_queue.get_nowait()
                        self.ready_queue.put(next_batch)
                except queue.Empty:
                    # No more immediately available batches
                    pass
            
            return batch
        except queue.Empty:
            if self.stop_event.is_set() and self.prefetch_queue.empty() and self.ready_queue.empty():
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
    
    # Adjust buffer size based on batch size and sequence length
    # This helps with memory stability and performance
    memory_per_batch = batch_size * max_length * 4  # Approx bytes per batch
    rec_buffer_size = min(128, max(16, int(1 * 1024 * 1024 * 1024 / memory_per_batch)))  # Target 1GB buffer
    
    # Use adjusted buffer size but keep within reasonable bounds
    actual_buffer_size = min(rec_buffer_size, buffer_size * 2)
    
    # Calculate optimal number of workers based on CPU cores and memory
    cpu_count = os.cpu_count() or 4
    optimal_workers = min(8, max(2, cpu_count // 2))
    
    print(f"Configuring data loader with {optimal_workers} workers, buffer size: {actual_buffer_size} groups")
    
    # Use Hugging Face dataset loader with optimized parameters
    return HuggingFaceDataLoader(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        buffer_size=actual_buffer_size,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_workers=optimal_workers,
        dataset_name=dataset_name,
        dataset_split=dataset_split
    )
