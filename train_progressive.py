#!/usr/bin/env python3
"""
Progressive Training Script for GPToughts
Automatically manages training on progressively longer sequences.

Usage:
    python train_progressive.py --model_type mla --size small --batch_size 8 \
                                --progressive_block_sizes 512,1024,2048,4096 \
                                --progressive_epochs_per_stage 5 \
                                --max_epochs 50 --out_dir out_progressive

Features:
- Automatic progression through multiple block sizes
- Checkpoint saving at each stage transition
- Automatic learning rate scaling
- Resume from interrupted progressive training
- Support for position interpolation on RoPE models
"""

import argparse
import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.lightning_module import LLMLightningModule
from data.data_loader_dynamic import create_dynamic_dataloader
from data.data_loader_packed import create_packed_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description='Progressive Training for GPToughts')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='mla', 
                       choices=['gpt', 'deepseek', 'llada', 'mla', 'mla_selective', 'parscale_mla', 'moe_mla', 'slm', 'nsa', 'hrm', 'mdm', 'sedd'],
                       help='Type of model to train')
    parser.add_argument('--size', type=str, default='small', 
                       choices=['tiny', 'small', 'medium', 'large', 'xl'],
                       help='Model size')
    
    # Progressive training specific
    parser.add_argument('--progressive_block_sizes', type=str, default='512,1024,2048,4096',
                       help='Comma-separated list of block sizes for progressive training')
    parser.add_argument('--progressive_epochs_per_stage', type=int, default=5,
                       help='Number of epochs to train at each stage')
    parser.add_argument('--progressive_lr_scale', type=float, default=0.7,
                       help='Learning rate scaling factor at each transition')
    parser.add_argument('--use_position_interpolation', action='store_true',
                       help='Use position interpolation for RoPE models')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--warmup_iters', type=int, default=2000, help='Warmup iterations')
    parser.add_argument('--lr_decay_iters', type=int, default=600000, help='LR decay iterations')
    parser.add_argument('--min_lr', type=float, default=6e-5, help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.95, help='Adam beta2')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--bias', action='store_true', help='Use bias in linear layers')
    parser.add_argument('--decay_lr', action='store_true', help='Use learning rate decay')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='apollo-mini', help='Dataset to use')
    parser.add_argument('--vocab_size', type=int, default=128256, help='Vocabulary size')
    parser.add_argument('--dataloader_type', type=str, default='dynamic', 
                       choices=['dynamic', 'packed'], help='Type of dataloader to use')
    
    # System arguments  
    parser.add_argument('--out_dir', type=str, default='out_progressive', help='Output directory')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--precision', type=str, default='bf16-mixed', help='Training precision')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Gradient accumulation')
    
    # Optimization arguments
    parser.add_argument('--optimizer_type', type=str, default='adamw', help='Optimizer type')
    parser.add_argument('--use_fp8', action='store_true', help='Use FP8 precision')
    parser.add_argument('--compile', action='store_true', help='Compile model with torch.compile')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='gptoughts-progressive', help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name')
    
    # Callbacks
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--save_top_k', type=int, default=3, help='Save top k checkpoints')
    
    return parser.parse_args()

def setup_progressive_args(args):
    """Setup progressive training arguments."""
    # Parse progressive block sizes
    if isinstance(args.progressive_block_sizes, str):
        args.progressive_block_sizes = [int(x.strip()) for x in args.progressive_block_sizes.split(',')]
    
    # Set initial block size to the first stage
    args.block_size = args.progressive_block_sizes[0]
    
    # Enable progressive training
    args.progressive_training = True
    
    return args

def create_dataloaders(args):
    """Create train and validation dataloaders."""
    print(f"Creating {args.dataloader_type} dataloaders with block_size={args.block_size}")
    
    if args.dataloader_type == 'dynamic':
        train_loader = create_dynamic_dataloader(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            block_size=args.block_size,
            num_workers=args.num_workers,
            split='train'
        )
        val_loader = create_dynamic_dataloader(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            block_size=args.block_size,
            num_workers=args.num_workers,
            split='validation'
        )
    elif args.dataloader_type == 'packed':
        train_loader = create_packed_dataloader(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            block_size=args.block_size,
            num_workers=args.num_workers,
            split='train'
        )
        val_loader = create_packed_dataloader(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            block_size=args.block_size,
            num_workers=args.num_workers,
            split='validation'
        )
    else:
        raise ValueError(f"Unknown dataloader type: {args.dataloader_type}")
    
    return train_loader, val_loader

def setup_callbacks(args):
    """Setup PyTorch Lightning callbacks."""
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.out_dir, 'checkpoints'),
        filename='{epoch}-{step}-{val_loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=args.save_top_k,
        save_last=True,
        every_n_epochs=1,
    )
    callbacks.append(checkpoint_callback)
    
    # Progressive training checkpoints
    progressive_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.out_dir, 'progressive_checkpoints'),
        filename='stage_{progressive_training/stage:d}_epoch_{epoch:d}',
        save_on_train_epoch_end=True,
        every_n_epochs=args.progressive_epochs_per_stage,
    )
    callbacks.append(progressive_checkpoint)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Early stopping
    if args.early_stopping_patience > 0:
        early_stop = EarlyStopping(
            monitor='val/loss',
            patience=args.early_stopping_patience,
            mode='min',
            verbose=True,
        )
        callbacks.append(early_stop)
    
    return callbacks

def setup_logger(args):
    """Setup logger."""
    if args.use_wandb:
        # Generate run name if not provided
        if args.wandb_run_name is None:
            block_sizes_str = '-'.join(map(str, args.progressive_block_sizes))
            args.wandb_run_name = f"{args.model_type}_{args.size}_progressive_{block_sizes_str}"
        
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
        return logger
    else:
        return True  # Default logger

class ProgressiveTrainingCallback(pl.Callback):
    """Callback to handle dynamic dataloader updates during progressive training."""
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.current_stage = 0
        
    def on_train_epoch_end(self, trainer, pl_module):
        """Check if we need to update dataloaders after stage transition."""
        if hasattr(pl_module, 'progressive_current_stage'):
            new_stage = pl_module.progressive_current_stage
            if new_stage != self.current_stage:
                self.current_stage = new_stage
                new_block_size = pl_module.progressive_block_sizes[new_stage]
                print(f"Updating dataloaders for new block_size: {new_block_size}")
                
                # Update args
                self.args.block_size = new_block_size
                
                # Create new dataloaders
                try:
                    train_loader, val_loader = create_dataloaders(self.args)
                    
                    # Replace dataloaders (this is a workaround - in practice you might 
                    # want to handle this more elegantly with a custom data module)
                    trainer.train_dataloader = train_loader
                    trainer.val_dataloaders = [val_loader]
                    
                    print(f"Successfully updated dataloaders for block_size {new_block_size}")
                    
                except Exception as e:
                    print(f"Warning: Failed to update dataloaders: {e}")
                    print("Continuing with existing dataloaders...")

def main():
    args = parse_args()
    args = setup_progressive_args(args)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'progressive_checkpoints'), exist_ok=True)
    
    print("=== Progressive Training Configuration ===")
    print(f"Model: {args.model_type} ({args.size})")
    print(f"Progressive block sizes: {args.progressive_block_sizes}")
    print(f"Epochs per stage: {args.progressive_epochs_per_stage}")
    print(f"Total estimated epochs: {len(args.progressive_block_sizes) * args.progressive_epochs_per_stage}")
    print(f"LR scale at transitions: {args.progressive_lr_scale}")
    print(f"Position interpolation: {args.use_position_interpolation}")
    print(f"Output directory: {args.out_dir}")
    print("=" * 42)
    
    # Create model
    model = LLMLightningModule(args)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)
    
    # Setup callbacks
    callbacks = setup_callbacks(args)
    callbacks.append(ProgressiveTrainingCallback(args))
    
    # Setup logger
    logger = setup_logger(args)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=args.out_dir,
        enable_progress_bar=True,
        log_every_n_steps=50,
        val_check_interval=0.25,  # Check validation 4 times per epoch
        enable_model_summary=True,
        reload_dataloaders_every_n_epochs=0,  # Disable automatic reloading since we handle it manually
    )
    
    # Print training info
    print("\n=== Training Info ===")
    print(f"Starting with block_size: {args.block_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Precision: {args.precision}")
    print("=" * 21)
    
    # Start training
    try:
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=args.resume_from_checkpoint
        )
        print("\n=== Progressive Training Completed Successfully ===")
        
        # Final model summary
        final_block_size = args.progressive_block_sizes[-1]
        final_stage = len(args.progressive_block_sizes) - 1
        print(f"Final block size: {final_block_size}")
        print(f"Total stages completed: {final_stage + 1}")
        print(f"Best validation loss: {trainer.callback_metrics.get('val/loss', 'N/A')}")
        
        # Save final model
        final_path = os.path.join(args.out_dir, 'final_progressive_model.ckpt')
        trainer.save_checkpoint(final_path)
        print(f"Final model saved to: {final_path}")
        
    except KeyboardInterrupt:
        print("\n=== Training Interrupted ===")
        print("Saving current state...")
        interrupt_path = os.path.join(args.out_dir, 'interrupted_progressive_model.ckpt')
        trainer.save_checkpoint(interrupt_path)
        print(f"Interrupted model saved to: {interrupt_path}")
        print("You can resume training with --resume_from_checkpoint")
        
    except Exception as e:
        print(f"\n=== Training Failed ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save model state for debugging
        try:
            error_path = os.path.join(args.out_dir, 'error_progressive_model.ckpt')
            trainer.save_checkpoint(error_path)
            print(f"Error model state saved to: {error_path}")
        except:
            print("Could not save error model state")
    
    finally:
        # Cleanup wandb
        if args.use_wandb:
            try:
                wandb.finish()
            except:
                pass

if __name__ == "__main__":
    main()