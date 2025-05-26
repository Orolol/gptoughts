import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import threading
import os
import time
from typing import Dict, List, Tuple, Optional
import numpy as np


class ConcatenatedDocumentBuffer:
    """Buffer that accumulates documents and creates concatenated sequences with document boundaries."""
    def __init__(self, capacity=4):
        self.capacity = capacity
        self.buffer = []
        self.lock = threading.Lock()
        self.not_full = threading.Condition(self.lock)
        self.not_empty = threading.Condition(self.lock)
        self.is_closed = False
    
    def put(self, batch):
        """Add a concatenated batch to the buffer."""
        with self.lock:
            while len(self.buffer) >= self.capacity and not self.is_closed:
                self.not_full.wait(timeout=1.0)
                if self.is_closed:
                    return False
            
            if self.is_closed:
                return False
                
            self.buffer.append(batch)
            self.not_empty.notify()
            return True
    
    def get(self):
        """Get a concatenated batch from the buffer."""
        with self.lock:
            while len(self.buffer) == 0 and not self.is_closed:
                self.not_empty.wait(timeout=1.0)
                if self.is_closed and len(self.buffer) == 0:
                    return None
            
            if len(self.buffer) == 0:
                return None
                
            batch = self.buffer.pop(0)
            self.not_full.notify()
            return batch
    
    def close(self):
        """Close the buffer and wake up all waiting threads."""
        with self.lock:
            self.is_closed = True
            self.not_empty.notify_all()
            self.not_full.notify_all()
    
    def __len__(self):
        with self.lock:
            return len(self.buffer)


class ConcatenatedDocumentDataset(IterableDataset):
    """
    Dataset that concatenates multiple documents into single long sequences.
    Each batch contains one long sequence with document boundary information for flex attention.
    """
    def __init__(
        self, 
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="CC-MAIN-2024-10",
        split='train', 
        max_length=8192,  # Much longer sequences
        documents_per_sequence=4,  # Number of documents to concatenate
        buffer_size=4,
        start_offset=0, 
        tokenizer=None,
        add_eos_between_docs=True,
        gradient_accumulation_steps=1
    ):
        super().__init__()
        
        # Load dataset
        if dataset_name == "HuggingFaceFW/fineweb-edu":
            self.dataset = load_dataset(
                dataset_name,
                name=dataset_config,
                split=split,
                streaming=True
            ).skip(start_offset)
        else:
            self.dataset = load_dataset(
                dataset_name,
                split=split,
                streaming=True
            ).skip(start_offset)
        
        # Initialize tokenizer
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
            
        self.max_length = max_length
        self.documents_per_sequence = documents_per_sequence
        self.buffer_size = buffer_size
        self.start_offset = start_offset
        self.add_eos_between_docs = add_eos_between_docs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Buffer for concatenated sequences
        self.batch_buffer = ConcatenatedDocumentBuffer(capacity=buffer_size)
        
        # Thread management
        self.prefetch_thread = None
        self.should_stop = threading.Event()
        
        # Statistics
        self.sequences_prepared = 0
        self.sequences_served = 0
        
        # Start prefetching
        self._start_prefetching()
        
        # Wait for first batch
        print("Initializing concatenated document data loader...")
        timeout = 30
        start_time = time.time()
        while len(self.batch_buffer) == 0 and time.time() - start_time < timeout:
            time.sleep(0.1)
        print(f"Data loader initialized, buffer has {len(self.batch_buffer)} sequences ready")

    def _concatenate_and_tokenize(self, documents: List[str]) -> Dict[str, torch.Tensor]:
        """
        Concatenate multiple documents and create document boundary information.
        
        Returns:
            Dict containing:
                - input_ids: concatenated token ids [1, seq_len]
                - attention_mask: standard attention mask [1, seq_len]
                - document_boundaries: list of (start, end) positions for each document
                - labels: shifted input_ids for autoregressive training [1, seq_len]
        """
        all_input_ids = []
        document_boundaries = []
        current_position = 0
        
        for i, doc in enumerate(documents):
            # Tokenize document
            tokens = self.tokenizer(
                doc,
                add_special_tokens=(i == 0),  # Only add special tokens for first doc
                truncation=False,
                return_attention_mask=False
            )['input_ids']
            
            # Record document boundary
            doc_start = current_position
            doc_end = current_position + len(tokens)
            document_boundaries.append((doc_start, doc_end))
            
            all_input_ids.extend(tokens)
            current_position = doc_end
            
            # Add EOS token between documents if requested
            if self.add_eos_between_docs and i < len(documents) - 1:
                all_input_ids.append(self.tokenizer.eos_token_id)
                current_position += 1
        
        # Truncate or pad to max_length
        if len(all_input_ids) > self.max_length:
            all_input_ids = all_input_ids[:self.max_length]
            # Update last document boundary if truncated
            if document_boundaries:
                last_start, _ = document_boundaries[-1]
                document_boundaries[-1] = (last_start, self.max_length)
        else:
            # Pad with EOS tokens
            padding_length = self.max_length - len(all_input_ids)
            all_input_ids.extend([self.tokenizer.eos_token_id] * padding_length)
        
        # Convert to tensors with batch dimension
        input_ids = torch.tensor(all_input_ids, dtype=torch.long).unsqueeze(0)  # [1, seq_len]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones_like(input_ids)
        if len(all_input_ids) < self.max_length:
            attention_mask[0, len(all_input_ids):] = 0
        
        # Create labels by shifting input_ids left by 1
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Ignore last position in loss
        
        # Set padding positions to -100 in labels
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids.contiguous(),
            'attention_mask': attention_mask.contiguous(),
            'labels': labels.contiguous(),
            'document_boundaries': document_boundaries,
            # For compatibility with existing training code
            'decoder_input_ids': input_ids.clone().contiguous(),
            'decoder_attention_mask': attention_mask.clone().contiguous(),
        }

    def _prefetch_data(self):
        """Background thread that continuously prefetches concatenated sequences."""
        try:
            dataset_iter = iter(self.dataset)
            
            while not self.should_stop.is_set():
                try:
                    # Collect documents for concatenation
                    documents = []
                    for _ in range(self.documents_per_sequence):
                        if self.should_stop.is_set():
                            break
                        
                        try:
                            example = next(dataset_iter)
                            # Handle different dataset formats
                            if 'text' in example:
                                documents.append(example['text'])
                            elif 'content' in example:
                                documents.append(example['content'])
                            else:
                                # Try to find any text field
                                for key in example:
                                    if isinstance(example[key], str):
                                        documents.append(example[key])
                                        break
                        except StopIteration:
                            # Dataset exhausted, restart
                            dataset_iter = iter(self.dataset)
                            example = next(dataset_iter)
                            if 'text' in example:
                                documents.append(example['text'])
                            elif 'content' in example:
                                documents.append(example['content'])
                    
                    if documents and not self.should_stop.is_set():
                        # Create concatenated sequence
                        batch = self._concatenate_and_tokenize(documents)
                        
                        # Add to buffer
                        if not self.batch_buffer.put(batch):
                            break
                        
                        self.sequences_prepared += 1
                        
                        if self.sequences_prepared % 100 == 0:
                            print(f"Prepared {self.sequences_prepared} concatenated sequences")
                    
                except StopIteration:
                    # Dataset fully exhausted
                    self.batch_buffer.close()
                    break
                
                except Exception as e:
                    print(f"Exception in prefetch thread: {e}")
                    self.batch_buffer.close()
                    break
                    
        except Exception as e:
            print(f"Fatal error in prefetch thread: {e}")
            self.batch_buffer.close()

    def _start_prefetching(self):
        """Start the prefetching thread."""
        if self.prefetch_thread is None or not self.prefetch_thread.is_alive():
            self.should_stop.clear()
            self.prefetch_thread = threading.Thread(
                target=self._prefetch_data, 
                daemon=True,
                name="ConcatenatedDataPrefetchThread"
            )
            self.prefetch_thread.start()
            print(f"Started concatenated document prefetch thread (id: {self.prefetch_thread.ident})")

    def __iter__(self):
        self._start_prefetching()
        return self
    
    def __next__(self):
        batch = self.batch_buffer.get()
        
        if batch is None:
            self.should_stop.set()
            raise StopIteration
        
        self.sequences_served += 1
        return batch

    def __del__(self):
        """Clean shutdown."""
        if hasattr(self, 'should_stop'):
            self.should_stop.set()
        
        if hasattr(self, 'batch_buffer'):
            self.batch_buffer.close()
            
        if hasattr(self, 'prefetch_thread') and self.prefetch_thread is not None:
            self.prefetch_thread.join(timeout=0.5)


def create_document_attention_mask(document_boundaries: List[Tuple[int, int]], seq_len: int) -> torch.Tensor:
    """
    Create an attention mask that respects document boundaries.
    Each document can only attend to tokens within the same document.
    
    Args:
        document_boundaries: List of (start, end) tuples for each document
        seq_len: Total sequence length
        
    Returns:
        attention_mask: [seq_len, seq_len] boolean tensor where True means "can attend"
    """
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool)
    
    for start, end in document_boundaries:
        # Each position in the document can attend to all positions in the same document
        # that come before it (causal within document)
        for i in range(start, end):
            for j in range(start, min(i + 1, end)):
                mask[i, j] = True
    
    return mask