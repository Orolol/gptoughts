import os
import sys
import torch
import subprocess

from data.openwebtext.data_loader import StreamingDataset

def get_gpu_count():
    return torch.cuda.device_count()

def get_datasets(block_size, batch_size, device):
    """Get train and validation datasets"""
    
    # Create base datasets
    train_dataset = StreamingDataset(
        block_size=block_size,
        batch_size=batch_size,
        split='train',
        device=device
    )
    
    val_dataset = StreamingDataset(
        block_size=block_size,
        batch_size=batch_size,
        split='train',
        device=device
    )
    
    # Determine optimal number of workers
    num_workers = min(8, os.cpu_count() // 2) if os.cpu_count() else 4
    
    # Create optimized DataLoaders
    train_loader = StreamingDataset.get_dataloader(
        train_dataset,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = StreamingDataset.get_dataloader(
        val_dataset,
        num_workers=2,  # Fewer workers for validation
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader

def main():
    world_size = get_gpu_count()
    if world_size == 0:
        print("No GPUs found. Running on CPU")
        subprocess.run([sys.executable, "train.py"])
        return
        
    print(f"Found {world_size} GPUs. Launching distributed training...")
    
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = "localhost"
    current_env["MASTER_PORT"] = "29500"
    current_env["WORLD_SIZE"] = str(world_size)
    
    processes = []
    for rank in range(world_size):
        current_env["RANK"] = str(rank)
        current_env["LOCAL_RANK"] = str(rank)
        
        cmd = [sys.executable, "train.py"]
        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)
    
    for process in processes:
        process.wait()
        if process.returncode != 0:
            print(f"Training failed with return code {process.returncode}")
            # Kill all remaining processes
            for p in processes:
                if p.poll() is None:  # If process is still running
                    p.kill()
            sys.exit(1)

if __name__ == "__main__":
    main() 