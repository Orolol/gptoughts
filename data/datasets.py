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
        
        # Enable fixed padding when using torch.compile for compatibility
        use_fixed_padding = getattr(args, 'compile', False)
        if use_fixed_padding:
            print("Note: Using fixed padding for torch.compile compatibility")

        train_dataset = DynamicFinewebDataset(
            split='train',
            max_length=args.block_size,
            buffer_size=args.batch_size * 4,
            shuffle=True,
            tokenizer=args.tokenizer,
            max_sequences_per_batch=args.batch_size,
            gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1),
            use_fixed_padding=use_fixed_padding,
            use_dynamic_batch_size=True,  # Enable dynamic batch sizing for optimal GPU usage
            dynamic_batch_safety_factor=getattr(args, 'dynamic_batch_safety_factor', 0.7),  # Conservative default
            dynamic_batch_max_multiplier=getattr(args, 'dynamic_batch_max_multiplier', 1.5)  # Max 1.5x base batch size
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

    elif dataloader_type == 'packed':
        from data.data_loader_packed import PackedFinewebDataset
        print("Using PackedFinewebDataset for training (fixed shape + document packing).")

        train_dataset = PackedFinewebDataset(
            split='train',
            max_length=args.block_size,
            batch_size=args.batch_size,
            tokenizer=args.tokenizer,
            prefetch_batches=16,
            buffer_docs=max(2048, args.batch_size * args.block_size),
            shuffle=True,
        )
        val_dataset = train_dataset

        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            collate_fn=lambda x: x[0]
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=0,
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
            num_workers=4,
            pin_memory=True,
            collate_fn=lambda x: x[0]
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1, # Batching is handled inside the dataset
            num_workers=4,
            pin_memory=True,
            collate_fn=lambda x: x[0]
        )

    return train_loader, val_loader
