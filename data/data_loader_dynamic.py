import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import threading
import random
import os
import time
from collections import deque
import heapq
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

class DynamicBatchBuffer:
    """Efficient buffer for dynamic batching with priority queue for size-based grouping"""
    
    def __init__(self, prefetch_size: int = 100, max_buffer_size: int = 500):
        self.prefetch_size = prefetch_size
        self.max_buffer_size = max_buffer_size
        self.tokenized_docs = []
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)
        self.is_closed = False
        self.total_docs_processed = 0
        
    def add_documents(self, documents: List[Dict[str, torch.Tensor]]):
        """Add tokenized documents to the buffer"""
        start_time = time.time()
        with self.lock:
            while len(self.tokenized_docs) >= self.max_buffer_size and not self.is_closed:
                self.not_full.wait(timeout=1.0)
                if self.is_closed:
                    return False
            
            if self.is_closed:
                return False
            
            self.tokenized_docs.extend(documents)
            self.total_docs_processed += len(documents)
            # Log very rarely or not at all during normal operation
            pass
            self.not_empty.notify_all()
            return True
    
    def get_batch(self, batch_size: int, max_tokens: Optional[int] = None, wait: bool = True) -> Optional[List[Dict[str, torch.Tensor]]]:
        """Get a batch of documents optimized for similar lengths
        
        Args:
            batch_size: Number of documents to get
            max_tokens: Maximum tokens per batch
            wait: If True, wait for documents. If False, return None immediately if not enough docs.
        """
        start_time = time.time()
        wait_logged = False
        with self.lock:
            # Non-blocking mode: return immediately if not enough documents
            if not wait and len(self.tokenized_docs) < batch_size:
                return None
            
            while len(self.tokenized_docs) < batch_size and not self.is_closed:
                # Only log during initial phase or when really stuck
                self.not_empty.wait(timeout=1.0)
                if self.is_closed and len(self.tokenized_docs) < batch_size:
                    # Return remaining documents if buffer is closing
                    if self.tokenized_docs:
                        batch = self.tokenized_docs[:]
                        self.tokenized_docs = []
                        self.not_full.notify()
                        return batch
                    return None
            
            if len(self.tokenized_docs) < batch_size:
                return None
            
            # Sort documents by length for better batching
            self.tokenized_docs.sort(key=lambda x: x['length'])
            
            # Select batch with similar lengths
            if max_tokens:
                # Dynamic batch size based on max tokens
                batch = []
                current_tokens = 0
                max_length_in_batch = 0
                
                for doc in self.tokenized_docs:
                    doc_length = doc['length']
                    # Update max length if this doc is added
                    new_max_length = max(max_length_in_batch, doc_length)
                    # Calculate total tokens if this doc is added
                    new_total_tokens = (len(batch) + 1) * new_max_length
                    
                    if new_total_tokens <= max_tokens and len(batch) < batch_size:
                        batch.append(doc)
                        max_length_in_batch = new_max_length
                        current_tokens = new_total_tokens
                    else:
                        break
                
                if not batch:
                    # If no documents fit, take at least one
                    batch = [self.tokenized_docs[0]]
                
                # Remove selected documents from buffer
                self.tokenized_docs = self.tokenized_docs[len(batch):]
            else:
                # Fixed batch size with similar lengths
                batch = self.tokenized_docs[:batch_size]
                self.tokenized_docs = self.tokenized_docs[batch_size:]
            
            self.not_full.notify()
            return batch
    
    def close(self):
        """Close the buffer"""
        with self.lock:
            self.is_closed = True
            self.not_empty.notify_all()
            self.not_full.notify_all()
    
    def __len__(self):
        with self.lock:
            return len(self.tokenized_docs)


class DynamicFinewebDataset(IterableDataset):
    """Dynamic batching dataset with efficient async prefetching and size-based grouping"""
    
    def __init__(
        self,
        split: str = 'train',
        max_length: int = 2048,
        max_sequences_per_batch: int = None,  # Matches datasets.py naming
        buffer_size: int = 16,  # Matches datasets.py naming
        prefetch_size: int = 100,
        max_tokens_per_batch: Optional[int] = None,
        shuffle: bool = True,
        start_offset: int = 0,
        tokenizer: Optional[Any] = None,
        gradient_accumulation_steps: int = 1,
        num_tokenizer_workers: int = 2,
        max_iterations: Optional[int] = None,  # Maximum iterations before stopping
        # Backward compatibility
        batch_size: int = None,  # Legacy parameter
        **kwargs
    ):
        super().__init__()
        
        # Handle backward compatibility for batch_size parameter
        if max_sequences_per_batch is None and batch_size is not None:
            max_sequences_per_batch = batch_size
        elif max_sequences_per_batch is None:
            max_sequences_per_batch = 4  # Default value
        
        # Dataset configuration
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="CC-MAIN-2024-10",
            split=split,
            streaming=True
        ).skip(start_offset)
        
        # Tokenizer setup
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            access_token = os.getenv('HF_TOKEN')
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-1B-Instruct",
                use_fast=True,
                access_token=access_token
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Batching parameters
        self.max_length = max_length
        self.batch_size = max_sequences_per_batch  # Internal batch size
        self.prefetch_size = prefetch_size
        self.max_buffer_size = buffer_size * max_sequences_per_batch  # Scale buffer by batch size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.shuffle = shuffle
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_workers = num_tokenizer_workers
        
        # Buffers and threading
        self.doc_buffer = DynamicBatchBuffer(prefetch_size, self.max_buffer_size)
        self.batch_queue = deque(maxlen=20)  # Ready batches - increased size
        self.batch_queue_lock = threading.Lock()
        self.batch_queue_not_empty = threading.Condition(self.batch_queue_lock)
        
        # Worker threads
        self.tokenizer_threads = []
        self.batch_builder_thread = None
        self.should_stop = threading.Event()
        self.exception = None
        
        # Iteration control
        self.max_iterations = max_iterations
        self.iteration_count = 0
        
        # Statistics
        self.stats = {
            'docs_tokenized': 0,
            'batches_created': 0,
            'batches_served': 0,
            'avg_padding_ratio': 0.0,
            'total_padding_tokens': 0,
            'total_tokens': 0
        }
        
        # Start workers
        # Simplified initialization message
        print(f"Initializing dynamic data loader (batch_size={self.batch_size}, max_length={self.max_length})")
        
        self._initialized = False
        self._workers_started = False
        
        # Start workers immediately for better startup time
        self._start_workers()
    
    def _tokenize_document(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a single document"""
        # Tokenize with truncation only (no padding yet)
        try:
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
                padding=False
            )
            
            input_ids = tokens['input_ids'].squeeze(0)
            actual_length = len(input_ids)
            
            return {
                'input_ids': input_ids,
                'length': actual_length,
                'text': text[:100]  # Keep snippet for debugging
            }
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error tokenizing document: {e}")
            raise
    
    def _tokenizer_worker(self, worker_id: int):
        """Worker thread that tokenizes documents"""
        try:
            dataset_iter = iter(self.dataset)
            
            local_buffer = []
            batch_count = 0
            docs_processed = 0
            
            while not self.should_stop.is_set():
                try:
                    # Wait if buffer is too full to avoid excessive memory usage
                    buffer_size = len(self.doc_buffer.tokenized_docs) if hasattr(self.doc_buffer, 'tokenized_docs') else 0
                    if buffer_size > self.max_buffer_size // 2:
                        if docs_processed == 0:  # Log only first time
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Worker {worker_id}: Buffer full, waiting...")
                        while buffer_size > self.max_buffer_size // 2 and not self.should_stop.is_set():
                            time.sleep(0.1)
                            buffer_size = len(self.doc_buffer.tokenized_docs) if hasattr(self.doc_buffer, 'tokenized_docs') else 0
                    
                    if self.should_stop.is_set():
                        break
                        
                    # Collect documents
                    collect_start = time.time()
                    docs_to_collect = self.prefetch_size // self.num_workers
                    
                    for i in range(docs_to_collect):
                        if self.should_stop.is_set():
                            break
                        
                        example = next(dataset_iter)
                        tokenized = self._tokenize_document(example['text'])
                        local_buffer.append(tokenized)
                        docs_processed += 1
                        
                        with self.batch_queue_lock:
                            self.stats['docs_tokenized'] += 1
                    
                    # Add to main buffer
                    if local_buffer and not self.should_stop.is_set():
                        if self.shuffle:
                            random.shuffle(local_buffer)
                        
                        add_result = self.doc_buffer.add_documents(local_buffer)
                        
                        if not add_result:
                            break  # Buffer closed
                        
                        batch_count += 1
                        local_buffer = []
                
                except StopIteration:
                    # Dataset exhausted - this is normal, just restart from beginning
                    if local_buffer:
                        self.doc_buffer.add_documents(local_buffer)
                        local_buffer = []
                    # Restart the dataset iterator for continuous streaming
                    dataset_iter = iter(self.dataset)
                    time.sleep(0.5)  # Brief pause before restarting
                
                except Exception as e:
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Error in tokenizer worker {worker_id}: {e}")
                    self.exception = e
                    break
        
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Fatal error in tokenizer worker {worker_id}: {e}")
            self.exception = e
        
        finally:
            # Worker finished - don't restart
            pass
    
    def _create_padded_batch(self, documents: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Create a padded batch from documents with dynamic padding"""
        start_time = time.time()
        if not documents:
            return None
        
        # Find max length in this batch (dynamic padding)
        max_len = max(doc['length'] for doc in documents)
        
        # Prepare tensors
        batch_size = len(documents)
        input_ids = torch.full((batch_size, max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        
        # Fill tensors
        for i, doc in enumerate(documents):
            doc_len = doc['length']
            input_ids[i, :doc_len] = doc['input_ids']
            attention_mask[i, :doc_len] = 1
        
        # Create labels for autoregressive training
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = self.tokenizer.pad_token_id
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        
        # Update padding statistics
        total_tokens = batch_size * max_len
        actual_tokens = sum(doc['length'] for doc in documents)
        padding_tokens = total_tokens - actual_tokens
        padding_ratio = padding_tokens / total_tokens if total_tokens > 0 else 0
        
        with self.batch_queue_lock:
            self.stats['total_tokens'] += total_tokens
            self.stats['total_padding_tokens'] += padding_tokens
            if self.stats['total_tokens'] > 0:
                self.stats['avg_padding_ratio'] = self.stats['total_padding_tokens'] / self.stats['total_tokens']
        
        # Track padding stats silently
        
        return {
            'input_ids': input_ids.contiguous(),
            'attention_mask': attention_mask.contiguous(),
            'decoder_input_ids': input_ids.clone().contiguous(),
            'decoder_attention_mask': attention_mask.clone().contiguous(),
            'labels': labels.contiguous()
        }
    
    def _batch_builder_worker(self):
        """Worker thread that builds batches from tokenized documents"""
        batch_count = 0
        last_status_time = time.time()
        consecutive_none = 0
        try:
            while not self.should_stop.is_set():
                # Don't build too many batches ahead
                queue_len = len(self.batch_queue)
                if queue_len >= self.batch_queue.maxlen - 2:
                    if batch_count == 0:  # Log only first time
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Batch builder: Queue full ({queue_len}/{self.batch_queue.maxlen}), waiting...")
                    while len(self.batch_queue) >= self.batch_queue.maxlen - 2 and not self.should_stop.is_set():
                        time.sleep(0.1)
                
                if self.should_stop.is_set():
                    break
                
                # Get documents for a batch (always wait for the first batch)
                docs = self.doc_buffer.get_batch(
                    self.batch_size,
                    self.max_tokens_per_batch,
                    wait=True  # Always wait for primary batch
                )
                
                if docs is None:
                    consecutive_none += 1
                    if consecutive_none > 10:  # Allow some failures before giving up
                        time.sleep(0.5)  # Wait longer if we're not getting docs
                    continue
                    
                # Successfully got documents
                consecutive_none = 0
                
                # Create padded batch
                batch = self._create_padded_batch(docs)
                
                if batch is not None:
                    # Build gradient accumulation group
                    batch_group = [batch]
                    
                    # Try to get additional batches for gradient accumulation (non-blocking)
                    for ga_idx in range(self.gradient_accumulation_steps - 1):
                        # Try to get additional batch without waiting
                        additional_docs = self.doc_buffer.get_batch(
                            self.batch_size,
                            self.max_tokens_per_batch,
                            wait=False  # Don't wait for additional batches
                        )
                        
                        if additional_docs:
                            additional_batch = self._create_padded_batch(additional_docs)
                            if additional_batch:
                                batch_group.append(additional_batch)
                        else:
                            break  # Use partial gradient accumulation
                    
                    # Add to batch queue
                    with self.batch_queue_lock:
                        self.batch_queue.append(batch_group)
                        self.stats['batches_created'] += len(batch_group)
                        batch_count += 1
                        self.batch_queue_not_empty.notify()
        
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Error in batch builder: {e}")
            self.exception = e
    
    def _start_workers(self):
        """Start all worker threads"""
        # Only start workers once
        if self._workers_started:
            return
        self._workers_started = True
        self.should_stop.clear()
        self.exception = None
        
        # Start tokenizer workers
        for i in range(self.num_workers):
            thread = threading.Thread(
                target=self._tokenizer_worker,
                args=(i,),
                daemon=True,
                name=f"TokenizerWorker-{i}"
            )
            thread.start()
            self.tokenizer_threads.append(thread)
        
        # Start batch builder
        self.batch_builder_thread = threading.Thread(
            target=self._batch_builder_worker,
            daemon=False,  # Make non-daemon to ensure proper cleanup
            name="BatchBuilder"
        )
        self.batch_builder_thread.start()
    
    def _wait_for_initial_batches(self):
        """Wait for initial batches to be ready"""
        timeout = 30  # Give more time for initial batches
        start_time = time.time()
        check_count = 0
        
        while time.time() - start_time < timeout:
            with self.batch_queue_lock:
                check_count += 1
                queue_size = len(self.batch_queue)
                
                if queue_size > 0:
                    # First batch ready
                    return
            
            # Check buffer size too
            buffer_size = len(self.doc_buffer) if hasattr(self, 'doc_buffer') else 0
            if buffer_size > 0 or queue_size > 0:
                # Some progress is being made
                time.sleep(0.5)
                continue
                
            if self.exception:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Exception detected: {self.exception}")
                raise self.exception
            
            time.sleep(0.1)
        
        # Timeout reached - continue anyway as workers might be slow to start
        # Don't print warnings in production
    
    def __iter__(self):
        """Return iterator"""
        # Start workers if not already started (backup in case __init__ didn't)
        if not self._workers_started:
            self._start_workers()
        
        # Wait for initial batches only on first real iteration
        if not self._initialized:
            self._wait_for_initial_batches()
            self._initialized = True
        
        self.current_batch_group = None
        self.current_batch_index = 0
        return self
    
    def __next__(self):
        """Get next batch"""
        # Check if we've reached max iterations
        if self.max_iterations is not None and self.iteration_count >= self.max_iterations:
            raise StopIteration
            
        # Check for exceptions
        if self.exception:
            raise self.exception
        
        # Return next batch from current group
        if self.current_batch_group and self.current_batch_index < len(self.current_batch_group):
            batch = self.current_batch_group[self.current_batch_index]
            self.current_batch_index += 1
            self.stats['batches_served'] += 1
            self.iteration_count += 1
            return batch
        
        # Get new batch group
        with self.batch_queue_lock:
            while len(self.batch_queue) == 0:
                if self.exception:
                    raise self.exception
                
                # Check if workers are done
                all_done = all(not t.is_alive() for t in self.tokenizer_threads)
                if all_done and len(self.batch_queue) == 0:
                    raise StopIteration
                
                self.batch_queue_not_empty.wait(timeout=1.0)
            
            self.current_batch_group = self.batch_queue.popleft()
        
        # Return first batch from new group
        self.current_batch_index = 1
        self.stats['batches_served'] += 1
        self.iteration_count += 1
        # Log progress occasionally
        if self.stats['batches_served'] % 1000 == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Served {self.stats['batches_served']} batches")
        return self.current_batch_group[0]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        with self.batch_queue_lock:
            return self.stats.copy()
    
    def close(self):
        """Clean shutdown"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Closing dynamic data loader")
        print(self.should_stop)
        if hasattr(self, 'should_stop'):
            stats = self.get_stats()
            print(f"\n{'='*60}")
            print(f"Data Loader Final Statistics:")
            print(f"  - Documents tokenized: {stats['docs_tokenized']:,}")
            print(f"  - Batches created: {stats['batches_created']:,}")
            print(f"  - Batches served: {stats['batches_served']:,}")
            print(f"  - Average padding ratio: {stats['avg_padding_ratio']:.2%}")
            print(f"{'='*60}\n")
            
            self.should_stop.set()
            
            # Give threads time to finish gracefully
            time.sleep(0.5)
            
            # Then close the buffer
            self.doc_buffer.close()
            
            # Wait for threads
            for i, thread in enumerate(self.tokenizer_threads):
                if thread.is_alive():
                    thread.join(timeout=0.5)
            
            if self.batch_builder_thread and self.batch_builder_thread.is_alive():
                self.batch_builder_thread.join(timeout=0.5)
    
    def __del__(self):
        """Destructor"""
        self.close()