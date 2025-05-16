"""
Point d'entrée pour l'entraînement des modèles LLM avec PyTorch Lightning.
Ce script configure et lance l'entraînement en utilisant le LLMLightningModule.
"""

import os
import sys
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, IterableDataset # Import necessary data components

# Import local modules
from train.lightning_module import LLMLightningModule
from optimization.cuda_optim import setup_cuda_optimizations, print_gpu_stats
from train.train_utils import find_latest_checkpoint, get_gpu_count

# --- Data Loading Logic (Adapted from legacy run_train) ---
# It's generally better to put this in a DataModule, but for simplicity, keep it here for now.
def get_datasets(block_size, batch_size, tokenizer=None, num_workers=4):
    """
    Retourne les DataLoaders d'entraînement et de validation.
    Utilise FinewebDataset si disponible, sinon fallback.
    """
    try:
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
            split='val', # Use validation split if available
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

    except ImportError:
        print("Warning: FinewebDataset not found. Falling back to simple random data.")
        from torch.utils.data import TensorDataset, DataLoader

        # Create dummy data
        train_data = torch.randint(0, tokenizer.vocab_size if tokenizer else 1000, (1000, block_size))
        val_data = torch.randint(0, tokenizer.vocab_size if tokenizer else 1000, (200, block_size))

        train_dataset = TensorDataset(train_data, train_data) # Input, Target
        val_dataset = TensorDataset(val_data, val_data)     # Input, Target

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM models with PyTorch Lightning')

    # Model Parameters
    parser.add_argument('--model_type', type=str, choices=['deepseek', 'llada', 'gpt'], default='gpt', help='Type of model to train')
    parser.add_argument('--size', type=str, choices=['small', 'medium', 'large'], default='small', help='Size of the model')

    # IO Parameters
    parser.add_argument('--output_dir', type=str, default='out_lightning', help='Output directory for checkpoints and logs')
    parser.add_argument('--init_from', type=str, default='scratch', choices=['scratch', 'resume'], help='Initialize from scratch or resume training')
    parser.add_argument('--resume_ckpt_path', type=str, default=None, help='Specific checkpoint path to resume from (overrides searching in output_dir)')
    parser.add_argument('--keep_checkpoints', type=int, default=3, help='Number of checkpoints to keep (-1 for all)')

    # Data Parameters
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size per device')
    parser.add_argument('--block_size', type=int, default=512, help='Context size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')

    # Model Config Parameters (passed to LightningModule)
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--bias', action='store_true', help='Use bias in linear layers')
    parser.add_argument('--attention_backend', type=str, default=None, help='Attention backend (e.g., flash)')

    # Optimizer Parameters (passed to LightningModule)
    parser.add_argument('--optimizer_type', type=str, default=None, choices=['adamw', 'lion', 'apollo', 'apollo-mini'], help='Optimizer type')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.95, help='Adam beta2')
    parser.add_argument('--decay_lr', action='store_true', default=True, help='Decay learning rate') # Default to True as common practice
    parser.add_argument('--warmup_iters', type=int, default=200, help='Warmup iterations')
    parser.add_argument('--lr_decay_iters', type=int, default=600000, help='LR decay iterations')
    parser.add_argument('--min_lr', type=float, default=3e-6, help='Minimum learning rate')

    # Training Parameters (for Lightning Trainer)
    parser.add_argument('--max_iters', type=int, default=100000, help='Maximum training iterations (steps)')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value (0 for no clipping)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--eval_interval_steps', type=int, default=1000, help='Validation interval in steps')
    parser.add_argument('--log_interval_steps', type=int, default=10, help='Logging interval in steps')
    parser.add_argument('--eval_only', action='store_true', help='Run only evaluation')
    parser.add_argument('--compile', action='store_true', help='Compile the model with torch.compile')

    # Precision Parameters (for Lightning Trainer)
    parser.add_argument('--precision', type=str, default='bf16-mixed', choices=['32-true', '16-mixed', 'bf16-mixed'], help='Training precision')
    # FP8 requires specific setup, handled separately if needed via transformer_engine integration within the module

    # Distributed Parameters (for Lightning Trainer)
    parser.add_argument('--strategy', type=str, default='ddp', help='Distributed strategy (e.g., ddp, fsdp)')
    parser.add_argument('--devices', type=int, default=-1, help='Number of GPUs to use (-1 for all available)')

    # MoE Parameters (passed to LightningModule)
    parser.add_argument('--router_z_loss_coef', type=float, default=0.001, help='Router loss coefficient')

    # BD3-LM Specific Args (passed to LightningModule)
    parser.add_argument('--use_bd3_training', action='store_true', help='Enable BD3-LM vectorized training path for LLaDA model')

    # Tokenizer Parameters
    parser.add_argument('--tokenizer_name', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help='Tokenizer name from Hugging Face Hub')

    # Logging
    parser.add_argument('--wandb_project', type=str, default=None, help='WandB project name (if None, uses TensorBoard)')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity name')

    # Advanced Optimizations (passed to LightningModule)
    parser.add_argument('--optimize_attention', action='store_true', help='Enable attention optimizations (if available)')
    parser.add_argument('--preallocate_memory', action='store_true', help='Preallocate CUDA memory (if available)')

    args = parser.parse_args()
    return args

# --- Main Execution ---
def main():
    args = parse_args()

    # Setup CUDA optimizations early
    if torch.cuda.is_available():
        setup_cuda_optimizations()
        if args.devices == -1: # Auto-detect GPUs if not specified
             args.devices = get_gpu_count()
        print(f"Detected {args.devices} GPUs.")
        if args.devices > 0:
             print_gpu_stats() # Print initial stats

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Tokenizer ---
    try:
        access_token = os.getenv('HF_TOKEN')
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True, access_token=access_token)
        tokenizer.pad_token = tokenizer.eos_token # Set pad token if needed
        args.vocab_size = tokenizer.vocab_size # Get vocab size from tokenizer
        args.tokenizer = tokenizer # Pass tokenizer via args for generation in module
        print(f"Initialized tokenizer: {args.tokenizer_name} with vocab size {args.vocab_size}")
    except Exception as e:
        print(f"Failed to load tokenizer '{args.tokenizer_name}': {e}")
        print("Proceeding without a tokenizer. Generation and some datasets might not work.")
        # Estimate vocab size or set a default if needed by model config
        args.vocab_size = getattr(args, 'vocab_size', 32000) # Use provided or default
        args.tokenizer = None

    # --- DataLoaders ---
    print("Setting up datasets...")
    train_loader, val_loader = get_datasets(
        args.block_size,
        args.batch_size,
        tokenizer=args.tokenizer,
        num_workers=args.num_workers
    )
    print("Datasets ready.")

    # --- LightningModule ---
    print("Initializing LightningModule...")
    model = LLMLightningModule(args)
    print("LightningModule initialized.")

    # --- Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='{epoch}-{step}-{val/loss:.2f}',
        save_top_k=args.keep_checkpoints,
        monitor='val/loss',
        mode='min',
        save_last=True, # Always save the last checkpoint
        every_n_train_steps=args.eval_interval_steps # Save checkpoint after validation
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    progress_bar = TQDMProgressBar(refresh_rate=10) # Adjust refresh rate as needed

    callbacks = [checkpoint_callback, lr_monitor, progress_bar]

    # --- Logger ---
    if args.wandb_project:
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            log_model=False, # Don't log model checkpoints to WandB by default
            save_dir=args.output_dir,
            config=vars(args) # Log hyperparameters
        )
        print(f"Using WandB logger (Project: {args.wandb_project})")
    else:
        logger = TensorBoardLogger(
            save_dir=args.output_dir,
            name="logs"
        )
        print(f"Using TensorBoard logger (Directory: {args.output_dir}/logs)")


    # --- Trainer ---
    # Determine checkpoint path for resuming
    ckpt_path = None
    if args.init_from == 'resume':
        if args.resume_ckpt_path:
            ckpt_path = args.resume_ckpt_path
            print(f"Resuming from specified checkpoint: {ckpt_path}")
        else:
            # Try to find the last checkpoint in the output directory
            ckpt_path = find_latest_checkpoint(args.output_dir, pattern="last.ckpt") # PL saves last as 'last.ckpt'
            if ckpt_path:
                print(f"Resuming from last checkpoint found: {ckpt_path}")
            else:
                print(f"Resume requested but no checkpoint found in {args.output_dir}. Starting from scratch.")
                args.init_from = 'scratch' # Fallback to scratch if no checkpoint

    # Configure Trainer
    trainer = pl.Trainer(
        devices=args.devices,
        accelerator="gpu" if torch.cuda.is_available() and args.devices != 0 else "cpu",
        strategy=args.strategy if args.devices > 1 else "auto",
        precision=args.precision,
        max_steps=args.max_iters,
        val_check_interval=args.eval_interval_steps, # Check validation every N steps
        check_val_every_n_epoch=None, # Disable epoch-based validation checking
        log_every_n_steps=args.log_interval_steps,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        # compile=args.compile, # torch.compile integration - enable if needed
        # deterministic=False, # For performance
        benchmark=True, # Enable cudnn benchmarking
        # limit_val_batches=100, # Limit validation batches if dataset is large
    )

    # Compile model if requested (do it after Trainer setup for FSDP compatibility)
    if args.compile:
        print("Compiling model with torch.compile...")
        try:
            # Note: Compilation might need to happen *after* strategy setup (e.g., FSDP)
            # It's often better to let the Trainer handle compilation if possible,
            # or compile within the LightningModule's setup hook based on the trainer strategy.
            # For now, compile directly here. Revisit if issues arise with strategies like FSDP.
            model = torch.compile(model, mode="max-autotune") # or reduce-overhead
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Model compilation failed: {e}")
            print("Proceeding without compilation.")


    # --- Start Training ---
    if args.eval_only:
        print("Starting evaluation only...")
        if not ckpt_path:
             print("Error: Evaluation only requested but no checkpoint specified or found.")
             sys.exit(1)
        trainer.validate(model, dataloaders=val_loader, ckpt_path=ckpt_path)
        print("Evaluation finished.")
    else:
        print("Starting training...")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
        print("Training finished.")

if __name__ == "__main__":
    main()