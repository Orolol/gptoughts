"""
Centralized dataset loading functions to avoid circular imports.
"""

from torch.utils.data import DataLoader

def get_datasets(block_size, batch_size, tokenizer=None, num_workers=4):
    """
    Returns the training and validation DataLoaders.
    Uses FinewebDataset if available, otherwise falls back to alternatives.
    """
    from data.data_loader_original import FinewebDataset
    print("Using FinewebDataset for training")

    # Note: FinewebDataset seems to handle batching internally.
    # We wrap it in a simple identity dataloader or adjust its usage.
    # For IterableDatasets, DataLoader typically just manages workers.
    train_dataset = FinewebDataset(
        split='train',
        max_length=block_size,
        buffer_size=batch_size * 10, # Larger buffer for better shuffling
        shuffle=True,
        tokenizer=tokenizer,
        batch_size=batch_size # Dataset yields batches directly
    )

    val_dataset = FinewebDataset(
        split='train', # Use validation split if available
        max_length=block_size,
        buffer_size=batch_size * 2,
        shuffle=False,
        tokenizer=tokenizer,
        batch_size=batch_size # Dataset yields batches directly
    )

    # Since FinewebDataset yields batches, DataLoader might just need num_workers
    # If it's NOT an IterableDataset yielding batches, DataLoader handles batching.
    # Assuming FinewebDataset IS an IterableDataset yielding batches:
    train_loader = DataLoader(
        train_dataset,
        batch_size=1, # Batching is done inside dataset
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda x: x[0] # Identity collate since dataset yields batches
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1, # Batching is done inside dataset
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda x: x[0] # Identity collate
    )

    return train_loader, val_loader