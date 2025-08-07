"""
Centralized dataset loading functions to avoid circular imports.
"""

from torch.utils.data import DataLoader

def get_datasets(args):
    """
    Returns the training and validation DataLoaders based on the specified dataloader type.
    """
    dataloader_type = getattr(args, 'dataloader_type', 'original')

    if dataloader_type == 'dynamic':
        from data.data_loader_dynamic import DynamicFinewebDataset
        print("Using FinewebDatasetDynamic (stable) for training.")

        train_dataset = DynamicFinewebDataset(
            split='train',
            max_length=args.block_size,
            buffer_size=args.batch_size * 4,
            shuffle=True,
            tokenizer=args.tokenizer,
            max_sequences_per_batch=args.batch_size,
            gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1)
            # No max_iterations for training - let Lightning handle epoch control
        )
        val_dataset = train_dataset

        # The dynamic loader uses internal threading, so we need num_workers=0
        train_loader = DataLoader(
            train_dataset,
            batch_size=1, # Batching is handled inside the dataset
            num_workers=0,  # Dynamic loader has its own threading
            pin_memory=True,
            collate_fn=lambda x: x[0]
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1, # Batching is handled inside the dataset
            num_workers=0,  # Dynamic loader has its own threading
            pin_memory=True,
            collate_fn=lambda x: x[0]
        )

    else: # 'original'
        from data.data_loader_original import FinewebDataset
        print("Using FinewebDataset (original) for training.")

        train_dataset = FinewebDataset(
            split='train',
            max_length=args.block_size,
            buffer_size=args.batch_size * 4,
            shuffle=True,
            tokenizer=args.tokenizer,
            batch_size=args.batch_size,
            gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1)
        )
        val_dataset = train_dataset
        
        # The original dataloader yields pre-formed batches, so we use an identity collate.
        train_loader = DataLoader(
            train_dataset,
            batch_size=1, # Batching is handled inside the dataset
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=lambda x: x[0]
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1, # Batching is handled inside the dataset
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=lambda x: x[0]
        )

    return train_loader, val_loader
