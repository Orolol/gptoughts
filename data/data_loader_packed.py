import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import threading
import os
import time
from typing import Dict, List, Tuple, Optional
import numpy as np


class PackedDocumentBuffer:
    """Buffer that accumulates packed document sequences."""
    def __init__(self, capacity=4):
        self.capacity = capacity
        self.buffer = []
        self.lock = threading.Lock()
        self.not_full = threading.Condition(self.lock)
        self.not_empty = threading.Condition(self.lock)
        self.is_closed = False
    
    def put(self, batch):
        """Add a packed batch to the buffer."""
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
        """Get a packed batch from the buffer."""
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


class PackedDocumentDataset(IterableDataset):
    """
    Dataset that packs multiple documents into sequences up to max_length.
    Documents are packed greedily - we keep adding documents until the next one won't fit.
    """
    def __init__(
        self, 
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="CC-MAIN-2024-10",
        split='train', 
        max_length=8192,
        buffer_size=4,
        start_offset=0, 
        tokenizer=None,
        add_eos_between_docs=True,
        min_length_ratio=0.8,  # Try to fill at least 80% of max_length
        gradient_accumulation_steps=1,
        tokenize_batch_size=10  # Number of documents to tokenize at once
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
        self.min_length = int(max_length * min_length_ratio)
        self.buffer_size = buffer_size
        self.start_offset = start_offset
        self.add_eos_between_docs = add_eos_between_docs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.tokenize_batch_size = tokenize_batch_size
        
        # Special tokens
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        
        # Buffer for packed sequences
        self.batch_buffer = PackedDocumentBuffer(capacity=buffer_size)
        
        # Thread management
        self.prefetch_thread = None
        self.should_stop = threading.Event()
        
        # Statistics
        self.sequences_prepared = 0
        self.sequences_served = 0
        self.documents_processed = 0
        self.total_tokens_packed = 0
        
        # Start prefetching
        self._start_prefetching()
        
        # Wait for first batch
        print("Initializing packed document data loader...")
        timeout = 30
        start_time = time.time()
        while len(self.batch_buffer) == 0 and time.time() - start_time < timeout:
            time.sleep(0.1)
        print(f"Data loader initialized, buffer has {len(self.batch_buffer)} sequences ready")

    def _tokenize_document(self, text: str) -> List[int]:
        """Tokenize a single document without padding."""
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False
        )['input_ids']
        return tokens

    def _pack_and_tokenize(self, documents: List[str], document_tokens: List[List[int]]) -> Dict[str, torch.Tensor]:
        """
        Pack multiple documents into a single sequence up to max_length.
        
        Args:
            documents: List of document texts
            document_tokens: List of tokenized documents (already tokenized for efficiency)
            
        Returns:
            Dict containing input_ids, labels, and document_boundaries
        """
        all_input_ids = []
        document_boundaries = []
        current_position = 0
        documents_included = 0
        
        # Add special tokens at the beginning
        all_input_ids.extend([self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else [])
        current_position = len(all_input_ids)
        
        for i, tokens in enumerate(document_tokens):
            # Check if adding this document would exceed max_length
            tokens_to_add = len(tokens)
            if self.add_eos_between_docs and i > 0:
                tokens_to_add += 1  # Account for EOS separator
            
            if current_position + tokens_to_add > self.max_length:
                # This document won't fit, stop packing
                break
            
            # Add EOS between documents if requested
            if self.add_eos_between_docs and i > 0:
                all_input_ids.append(self.eos_token_id)
                current_position += 1
            
            # Add document tokens
            doc_start = current_position
            all_input_ids.extend(tokens)
            current_position += len(tokens)
            doc_end = current_position
            
            # Record document boundary
            document_boundaries.append((doc_start, doc_end))
            documents_included += 1
        
        # Pad to max_length if needed
        if len(all_input_ids) < self.max_length:
            padding_length = self.max_length - len(all_input_ids)
            all_input_ids.extend([self.pad_token_id] * padding_length)
        
        # Convert to tensors with batch dimension
        input_ids = torch.tensor(all_input_ids[:self.max_length], dtype=torch.long).unsqueeze(0)  # [1, seq_len]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones_like(input_ids)
        if current_position < self.max_length:
            attention_mask[0, current_position:] = 0
        
        # Create labels by shifting input_ids left by 1
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Ignore last position in loss
        
        # Set padding positions to -100 in labels
        labels[attention_mask == 0] = -100
        
        # Pre-compute document mask for attention
        # This creates a 4D mask [B=1, 1, S, T] where attention is blocked across document boundaries
        seq_len = self.max_length
        document_mask = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
        
        # Start with causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        document_mask[0, 0] = -causal_mask * 1e10  # Large negative value for positions to mask
        
        # Add document boundary constraints
        for doc_start, doc_end in document_boundaries:
            # For each position in this document, it can only attend to positions within the same document
            for pos in range(doc_start, doc_end):
                # Mask out positions before this document
                if doc_start > 0:
                    document_mask[0, 0, pos, :doc_start] = -1e10
                # Mask out positions after this document
                if doc_end < seq_len:
                    document_mask[0, 0, pos, doc_end:] = -1e10
        
        # Mask out padding positions
        if current_position < seq_len:
            document_mask[0, 0, :, current_position:] = -1e10
            document_mask[0, 0, current_position:, :] = -1e10
        
        # Statistics
        utilization = current_position / self.max_length
        
        return {
            'input_ids': input_ids.contiguous(),
            'attention_mask': attention_mask.contiguous(),
            'labels': labels.contiguous(),
            'document_boundaries': document_boundaries,
            'document_mask': document_mask.contiguous(),  # Pre-computed 4D mask
            'documents_packed': documents_included,
            'tokens_used': current_position,
            'utilization': utilization,
            # For compatibility with existing training code
            'decoder_input_ids': input_ids.clone().contiguous(),
            'decoder_attention_mask': attention_mask.clone().contiguous(),
        }

    def _prefetch_data(self):
        """Background thread that continuously prefetches and packs documents."""
        try:
            dataset_iter = iter(self.dataset)
            
            # Buffer for documents waiting to be packed
            document_buffer = []
            tokenized_buffer = []
            
            while not self.should_stop.is_set():
                try:
                    # Collect documents until we have enough to potentially fill a sequence
                    while len(tokenized_buffer) < self.tokenize_batch_size and not self.should_stop.is_set():
                        try:
                            example = next(dataset_iter)
                            # Extract text from example
                            text = None
                            if 'text' in example:
                                text = example['text']
                            elif 'content' in example:
                                text = example['content']
                            else:
                                # Try to find any text field
                                for key in example:
                                    if isinstance(example[key], str):
                                        text = example[key]
                                        break
                            
                            if text:
                                # Tokenize the document
                                tokens = self._tokenize_document(text)
                                # Only keep documents that aren't too long
                                if len(tokens) <= self.max_length - 100:  # Leave some room for special tokens
                                    document_buffer.append(text)
                                    tokenized_buffer.append(tokens)
                                    self.documents_processed += 1
                                
                        except StopIteration:
                            # Dataset exhausted, restart
                            dataset_iter = iter(self.dataset)
                    
                    # Try to pack documents into sequences
                    while len(tokenized_buffer) > 0 and not self.should_stop.is_set():
                        # Greedily pack documents
                        packed_docs = []
                        packed_tokens = []
                        current_length = 1  # Account for BOS token
                        
                        i = 0
                        while i < len(tokenized_buffer):
                            doc_length = len(tokenized_buffer[i])
                            if packed_docs and self.add_eos_between_docs:
                                doc_length += 1  # Account for EOS separator
                            
                            if current_length + doc_length <= self.max_length:
                                # This document fits
                                packed_docs.append(document_buffer[i])
                                packed_tokens.append(tokenized_buffer[i])
                                current_length += doc_length
                                # Remove from buffers
                                document_buffer.pop(i)
                                tokenized_buffer.pop(i)
                            else:
                                # Try next document
                                i += 1
                        
                        # Create batch if we have enough content
                        if current_length >= self.min_length or (len(tokenized_buffer) == 0 and packed_docs):
                            batch = self._pack_and_tokenize(packed_docs, packed_tokens)
                            
                            # Add to buffer
                            if not self.batch_buffer.put(batch):
                                break
                            
                            self.sequences_prepared += 1
                            self.total_tokens_packed += batch['tokens_used']
                            
                            if self.sequences_prepared % 100 == 0:
                                avg_utilization = self.total_tokens_packed / (self.sequences_prepared * self.max_length)
                                print(f"Prepared {self.sequences_prepared} packed sequences "
                                      f"({self.documents_processed} docs, "
                                      f"{avg_utilization:.1%} avg utilization)")
                        else:
                            # Not enough content, wait for more documents
                            break
                    
                except Exception as e:
                    print(f"Exception in prefetch thread: {e}")
                    import traceback
                    traceback.print_exc()
                    self.batch_buffer.close()
                    break
                    
        except Exception as e:
            print(f"Fatal error in prefetch thread: {e}")
            import traceback
            traceback.print_exc()
            self.batch_buffer.close()

    def _start_prefetching(self):
        """Start the prefetching thread."""
        if self.prefetch_thread is None or not self.prefetch_thread.is_alive():
            self.should_stop.clear()
            self.prefetch_thread = threading.Thread(
                target=self._prefetch_data, 
                daemon=True,
                name="PackedDataPrefetchThread"
            )
            self.prefetch_thread.start()
            print(f"Started packed document prefetch thread (id: {self.prefetch_thread.ident})")

    def __iter__(self):
        self._start_prefetching()
        return self
    
    def __next__(self):
        batch = self.batch_buffer.get()
        
        if batch is None:
            self.should_stop.set()
            raise StopIteration
        
        self.sequences_served += 1
        
        # Log statistics occasionally
        if self.sequences_served % 100 == 0:
            print(f"Served {self.sequences_served} sequences, "
                  f"latest: {batch['documents_packed']} docs, "
                  f"{batch['utilization']:.1%} utilization")
        
        return batch

    def __del__(self):
        """Clean shutdown."""
        if hasattr(self, 'should_stop'):
            self.should_stop.set()
        
        if hasattr(self, 'batch_buffer'):
            self.batch_buffer.close()
            
        if hasattr(self, 'prefetch_thread') and self.prefetch_thread is not None:
            self.prefetch_thread.join(timeout=0.5)