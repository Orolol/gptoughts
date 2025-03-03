import os
import sys
import torch
import subprocess

from data_loader import FinewebDataset
from train_utils import get_gpu_count, setup_cuda_optimizations

def get_datasets(block_size, batch_size, device, tokenizer=None):
    """
    Return train and validation datasets
    """
    train_dataset = FinewebDataset(
        split='train',
        max_length=block_size,
        buffer_size=batch_size * 4,
        shuffle=True,
        tokenizer=tokenizer
    )
    
    return train_dataset

def launch_distributed_training(world_size):
    """
    Launch distributed training with the specified number of GPUs
    Args:
        world_size: Number of GPUs to use
    """
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
    
    return processes

def monitor_processes(processes):
    """
    Monitor training processes and handle failures
    Args:
        processes: List of subprocess.Popen objects to monitor
    Returns:
        bool: True if all processes completed successfully, False otherwise
    """
    for process in processes:
        process.wait()
        if process.returncode != 0:
            print(f"Training failed with return code {process.returncode}")
            # Kill all remaining processes
            for p in processes:
                if p.poll() is None:  # If process is still running
                    p.kill()
            return False
    return True

def main():
    # Initialize CUDA optimizations
    if torch.cuda.is_available():
        setup_cuda_optimizations()
    
    # Get number of available GPUs
    world_size = get_gpu_count()
    
    if world_size == 0:
        print("No GPUs found. Running on CPU")
        subprocess.run([sys.executable, "train.py"])
        return
    
    # Launch distributed training
    processes = launch_distributed_training(world_size)
    
    # Monitor processes
    success = monitor_processes(processes)
    if not success:
        sys.exit(1)
    
    print("Training completed successfully")

if __name__ == "__main__":
    main() 