import os
import sys
import torch
import subprocess

from data.openwebtext.data_loader import StreamingDataset

def get_gpu_count():
    return torch.cuda.device_count()

def get_datasets(block_size, batch_size, device):
    """
    Retourne les datasets (train + val) déjà initialisés.
    """
    train_dataset = StreamingDataset(
        block_size=block_size,
        batch_size=batch_size,
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="CC-MAIN-2024-10",
        split="train",
        device=device
    )
    
    val_dataset = StreamingDataset(
        block_size=block_size,
        batch_size=batch_size,
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="CC-MAIN-2024-10",
        split="train",
        device=device
    )
    return train_dataset, val_dataset

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