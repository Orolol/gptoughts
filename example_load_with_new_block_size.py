#!/usr/bin/env python3
"""
Example: Loading a model trained on one block_size and using it with a different block_size

This demonstrates how to:
1. Load a model checkpoint trained on block_size=1024
2. Adapt it to work with block_size=2048
3. Continue training or perform inference

Usage:
    python example_load_with_new_block_size.py \
        --checkpoint_path out/checkpoints/epoch-10-step-5000.ckpt \
        --original_block_size 1024 \
        --new_block_size 2048 \
        --model_type mla \
        --size small
"""

import argparse
import torch
from train.lightning_module import LLMLightningModule

def parse_args():
    parser = argparse.ArgumentParser(description='Load model with different block size')
    
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to the checkpoint to load')
    parser.add_argument('--original_block_size', type=int, required=True,
                       help='Original block size the model was trained on')
    parser.add_argument('--new_block_size', type=int, required=True,
                       help='New block size to adapt the model to')
    parser.add_argument('--model_type', type=str, default='mla',
                       help='Model type')
    parser.add_argument('--size', type=str, default='small',
                       help='Model size')
    parser.add_argument('--vocab_size', type=int, default=128256,
                       help='Vocabulary size')
    parser.add_argument('--use_position_interpolation', action='store_true',
                       help='Use position interpolation for RoPE models')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save the adapted model (optional)')
    
    return parser.parse_args()

def create_args_for_model(original_block_size, new_block_size, model_type, size, vocab_size, use_pi, checkpoint_path):
    """Create arguments object for the model."""
    import argparse
    
    # Create a proper argparse Namespace object
    args = argparse.Namespace()
    
    # Model config
    args.model_type = model_type
    args.size = size
    args.vocab_size = vocab_size
    args.block_size = new_block_size  # Target block size
    args.dropout = 0.1
    args.bias = False
    
    # Training config (not used but needed for initialization)
    args.learning_rate = 6e-4
    args.weight_decay = 0.01
    args.beta1 = 0.9
    args.beta2 = 0.95
    args.grad_clip = 1.0
    args.warmup_iters = 2000
    args.lr_decay_iters = 600000
    args.min_lr = 6e-5
    args.decay_lr = True
    args.batch_size = 8
    
    # Optimization
    args.use_fp8 = False
    args.optimizer_type = 'adamw'
    args.compile = False
    
    # Logging
    args.use_wandb = False
    
    # Progressive training (disabled)
    args.progressive_training = False
    
    # Block size adaptation
    args.load_checkpoint_path = checkpoint_path
    args.original_block_size = original_block_size
    args.use_position_interpolation = use_pi
    
    return args

def main():
    args = parse_args()
    
    print("=== Loading Model with New Block Size ===")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Original block size: {args.original_block_size}")
    print(f"New block size: {args.new_block_size}")
    print(f"Model: {args.model_type} ({args.size})")
    print(f"Position interpolation: {args.use_position_interpolation}")
    print("=" * 42)
    
    # Create model args
    model_args = create_args_for_model(
        args.original_block_size,
        args.new_block_size,
        args.model_type,
        args.size,
        args.vocab_size,
        args.use_position_interpolation,
        args.checkpoint_path
    )
    
    # Create and load model
    print("Creating model and loading checkpoint...")
    try:
        model = LLMLightningModule(model_args)
        print("✓ Model loaded and adapted successfully!")
        
        # Verify the model works with the new block size
        print(f"\nTesting model with new block_size={args.new_block_size}...")
        
        # Create dummy input
        batch_size = 2
        dummy_input = torch.randint(0, args.vocab_size, (batch_size, args.new_block_size))
        dummy_targets = torch.randint(0, args.vocab_size, (batch_size, args.new_block_size))
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input, targets=dummy_targets)
            
        print(f"✓ Forward pass successful!")
        print(f"  Logits shape: {outputs['logits'].shape}")
        print(f"  Loss: {outputs['loss'].item():.4f}")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"\nModel info:")
        print(f"  Total parameters: {total_params:.2f}M")
        print(f"  Trainable parameters: {trainable_params:.2f}M")
        print(f"  Current block_size: {model.args.block_size}")
        
        # Save adapted model if requested
        if args.output_path:
            print(f"\nSaving adapted model to: {args.output_path}")
            
            # Load the original checkpoint to get metadata
            try:
                original_checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
            except Exception as e:
                print(f"Warning: Could not load original checkpoint metadata: {e}")
                original_checkpoint = {}
            
            # Check if original model was compiled
            original_state_dict = original_checkpoint.get('state_dict', {})
            was_compiled = any(key.startswith('_orig_mod.') or key.startswith('model._orig_mod.') for key in original_state_dict.keys())
            
            print(f"Original model compilation status: {'compiled' if was_compiled else 'not compiled'}")
            
            # Get current model state dict
            current_model_state = model.model.state_dict()
            
            # Create the state dict in the correct format
            if was_compiled:
                # If original was compiled, save with _orig_mod prefix
                state_dict = {f'model._orig_mod.{k}': v for k, v in current_model_state.items()}
                print("Saving with compiled model format (_orig_mod prefix)")
            else:
                # If original was not compiled, save without prefix
                state_dict = {f'model.{k}': v for k, v in current_model_state.items()}
                print("Saving with non-compiled model format")
            
            # Create a complete Lightning checkpoint preserving original metadata
            lightning_checkpoint = {
                'state_dict': state_dict,
                'epoch': original_checkpoint.get('epoch', 0),  # Preserve original epoch
                'global_step': original_checkpoint.get('global_step', 0),  # Preserve original step
                'pytorch-lightning_version': original_checkpoint.get('pytorch-lightning_version', '2.0.0'),
                'state_dict_version': original_checkpoint.get('state_dict_version', '1.0'),
                
                # Preserve original hyperparameters and update relevant ones
                'hyper_parameters': {
                    **original_checkpoint.get('hyper_parameters', {}),
                    'block_size': args.new_block_size,  # Updated block size
                    'adapted_from_block_size': args.original_block_size,  # Track adaptation
                    'use_position_interpolation': args.use_position_interpolation,
                    'was_compiled': was_compiled,  # Track compilation status
                },
                
                # Preserve optimizer and scheduler states if they exist
                'optimizer_states': original_checkpoint.get('optimizer_states', []),
                'lr_schedulers': original_checkpoint.get('lr_schedulers', []),
                
                # Preserve other Lightning metadata
                'epoch': original_checkpoint.get('epoch', 0),
                'global_step': original_checkpoint.get('global_step', 0),
                'train_dataloader_config_id': original_checkpoint.get('train_dataloader_config_id'),
                'val_dataloader_config_id': original_checkpoint.get('val_dataloader_config_id'),
                'test_dataloader_config_id': original_checkpoint.get('test_dataloader_config_id'),
            }
            
            # Remove None values
            lightning_checkpoint = {k: v for k, v in lightning_checkpoint.items() if v is not None}
            
            torch.save(lightning_checkpoint, args.output_path)
            print("✓ Complete Lightning checkpoint saved!")
            
            # Check file sizes
            try:
                import os
                original_size = os.path.getsize(args.checkpoint_path) / (1024*1024)  # MB
                new_size = os.path.getsize(args.output_path) / (1024*1024)  # MB
                print(f"Original checkpoint: {original_size:.1f} MB")
                print(f"Adapted checkpoint: {new_size:.1f} MB")
                if new_size < original_size * 0.8:  # Less than 80% of original
                    print("⚠️ Warning: Adapted checkpoint is significantly smaller. Some data may be missing.")
                else:
                    print("✓ Checkpoint size preserved")
            except Exception as e:
                print(f"Could not compare file sizes: {e}")
            
            # Also save just the model weights for other uses
            weights_path = args.output_path.replace('.ckpt', '_weights.pth')
            torch.save(current_model_state, weights_path)
            print(f"✓ Model weights also saved to: {weights_path}")
            
        print(f"\n=== Adaptation Completed Successfully ===")
        print(f"Your model is now ready to work with block_size={args.new_block_size}")
        
        # Usage examples
        print(f"\nTo continue training with the adapted model:")
        print(f"python run_train.py --model_type {args.model_type} --size {args.size} \\")
        print(f"                    --block_size {args.new_block_size} \\")
        print(f"                    --load_checkpoint_path {args.checkpoint_path} \\")
        print(f"                    --original_block_size {args.original_block_size}")
        
        if args.use_position_interpolation:
            print(f"                    --use_position_interpolation")
            
    except Exception as e:
        print(f"✗ Error during model loading/adaptation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())