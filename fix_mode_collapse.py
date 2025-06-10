#!/usr/bin/env python3
"""
Script to fix a checkpoint that has fallen into mode collapse.
This script can help recover training by resetting certain parameters.
"""

import torch
import argparse
import os
from pathlib import Path

def fix_checkpoint(checkpoint_path, output_path=None):
    """
    Fix a checkpoint that has fallen into mode collapse.
    
    Args:
        checkpoint_path: Path to the checkpoint to fix
        output_path: Where to save the fixed checkpoint (default: overwrites original)
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get the state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Reset output projection layers which might be stuck
    # These are typically the layers that directly produce token predictions
    reset_count = 0
    noise_scale = 0.02  # Small noise to break symmetry
    
    for name, param in state_dict.items():
        # Reset output projections and final layers
        if any(pattern in name for pattern in ['lm_head', 'output_proj', 'final_proj', 'head']):
            print(f"Resetting layer: {name}")
            # Add small noise to break the symmetry
            if param.dim() >= 2:
                # Xavier/He initialization with small noise
                torch.nn.init.xavier_normal_(param)
                param.data += torch.randn_like(param) * noise_scale
            else:
                # For bias or 1D params
                param.data.fill_(0.0)
                param.data += torch.randn_like(param) * noise_scale
            reset_count += 1
    
    # Optionally reset router parameters if using MoE
    for name, param in state_dict.items():
        if 'router' in name and 'weight' in name:
            print(f"Adding noise to router: {name}")
            param.data += torch.randn_like(param) * noise_scale * 0.5  # Smaller noise for routers
            reset_count += 1
    
    # Reset optimizer state if present (this is important!)
    if 'optimizer' in checkpoint:
        print("Resetting optimizer state...")
        # Clear momentum/variance estimates that might be stuck
        optimizer_state = checkpoint['optimizer']
        if 'state' in optimizer_state:
            for state in optimizer_state['state'].values():
                if 'exp_avg' in state:
                    state['exp_avg'].zero_()  # Reset momentum
                if 'exp_avg_sq' in state:
                    # Don't fully reset variance, just reduce it
                    state['exp_avg_sq'].mul_(0.1)
    
    # Increase learning rate slightly to help escape local minimum
    if 'optimizer' in checkpoint and 'param_groups' in checkpoint['optimizer']:
        for group in checkpoint['optimizer']['param_groups']:
            if 'lr' in group:
                old_lr = group['lr']
                # Increase LR by 2x temporarily
                group['lr'] = old_lr * 2.0
                print(f"Increased learning rate from {old_lr} to {group['lr']}")
    
    # Save the fixed checkpoint
    if output_path is None:
        output_path = checkpoint_path.replace('.ckpt', '_fixed.ckpt')
    
    print(f"Saving fixed checkpoint to {output_path}")
    torch.save(checkpoint, output_path)
    print(f"Fixed {reset_count} parameters")
    print("Done! You can resume training with the fixed checkpoint.")
    print("\nRecommendations:")
    print("1. Monitor the loss carefully - it should increase initially then stabilize")
    print("2. Reduce learning rate back to normal after ~100-200 steps")
    print("3. Watch for diverse text generation after resuming")

def main():
    parser = argparse.ArgumentParser(description='Fix a checkpoint with mode collapse')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--output', type=str, default=None, help='Output path (default: adds _fixed suffix)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    fix_checkpoint(args.checkpoint, args.output)

if __name__ == '__main__':
    main()