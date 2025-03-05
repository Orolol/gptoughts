
import torch
from datasets import load_dataset

class StreamingBatchLoader(torch.utils.data.IterableDataset):
    """Simple streaming data loader that always keeps one tokenized batch ready.
    Works with HuggingFace datasets in streaming mode and returns batches of 
    size batch_size * gradient_accumulation_steps."""
    
    def __init__(self, dataset_name, dataset_split, tokenizer, batch_size, gradient_accumulation_steps, 
                 max_length=1024, text_column="text"):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_length = max_length
        self.text_column = text_column
        self.worker_id = 0
        self.num_workers = 0
    
    def _tokenize_function(self, examples):
        return self.tokenizer(
            examples[self.text_column],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
    
    def _get_stream(self):
        # Each worker will get its own stream
        return load_dataset(self.dataset_name, split=self.dataset_split, streaming=True)
    
    def __iter__(self):
        # Set up worker info for distributed loading
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        
        # Get a stream of data
        stream = self._get_stream()
        stream_iter = iter(stream)
        
        # Process the stream one batch at a time
        while True:
            # Collect samples for a single batch (not the full gradient accumulation batch)
            batch_samples = []
            for _ in range(self.batch_size):
                try:
                    sample = next(stream_iter)
                    batch_samples.append(sample[self.text_column])
                except StopIteration:
                    # Restart dataset if we reached the end during batch collection
                    stream_iter = iter(self._get_stream())
                    sample = next(stream_iter)
                    batch_samples.append(sample[self.text_column])
            
            # Tokenize the collected samples
            if not batch_samples:
                continue
                
            tokenized_batch = self._tokenize_function({"text": batch_samples})
            input_ids = tokenized_batch["input_ids"]
            
            # Yield x and y (for autoregressive prediction they are the same)
            yield input_ids, input_ids
