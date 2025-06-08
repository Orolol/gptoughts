#!/usr/bin/env python3
"""Fix checkpoint with mismatched positional embedding size."""

import torch
import sys

def fix_checkpoint(checkpoint_path, target_block_size):
    """Fix positional embedding size in checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Check current positional embedding
    pos_emb_key = None
    for k in ckpt['state_dict'].keys():
        if 'pos_emb.weight' in k:
            pos_emb_key = k
            break
    
    if pos_emb_key is None:
        print("No positional embedding found in checkpoint!")
        return
    
    current_shape = ckpt['state_dict'][pos_emb_key].shape
    print(f"Current positional embedding shape: {current_shape}")
    
    if current_shape[0] == target_block_size:
        print(f"Positional embedding already has correct size {target_block_size}")
        return
    
    # Truncate or pad the positional embedding
    if current_shape[0] > target_block_size:
        print(f"Truncating positional embedding from {current_shape[0]} to {target_block_size}")
        ckpt['state_dict'][pos_emb_key] = ckpt['state_dict'][pos_emb_key][:target_block_size]
    else:
        print(f"ERROR: Current size {current_shape[0]} is smaller than target {target_block_size}")
        print("Cannot fix this checkpoint - would need to expand positional embeddings")
        return
    
    # Update hyperparameters if present
    if 'hyper_parameters' in ckpt:
        ckpt['hyper_parameters']['block_size'] = target_block_size
        print(f"Updated block_size in hyperparameters to {target_block_size}")
    
    # Save fixed checkpoint
    output_path = checkpoint_path.replace('.ckpt', '_fixed.ckpt')
    print(f"Saving fixed checkpoint to {output_path}...")
    torch.save(ckpt, output_path)
    print("Done!")
    
    return output_path

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_checkpoint.py <checkpoint_path> <target_block_size>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    target_block_size = int(sys.argv[2])
    
    fix_checkpoint(checkpoint_path, target_block_size)