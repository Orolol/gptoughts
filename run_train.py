import os
import sys
import torch
import subprocess
from data.openwebtext.data_loader import StreamingDataset
from torch.utils.data.distributed import DistributedSampler

def get_gpu_count():
    return torch.cuda.device_count()

def init_dataset():
    """Initialize dataset once and save it to disk for all processes to use"""
    if not os.path.exists('data/shared_dataset'):
        os.makedirs('data/shared_dataset', exist_ok=True)
        
    dataset_path = 'data/shared_dataset/dataset.pt'
    
    if not os.path.exists(dataset_path):
        print("Initializing shared dataset...")
        dataset = StreamingDataset(
            block_size=64,  # Ces valeurs seront écrasées lors de l'utilisation
            batch_size=32,
            dataset_name="HuggingFaceFW/fineweb-2",
            dataset_config="fra_Latn",
            split="train",
            device='cpu'  # Initialiser sur CPU d'abord
        )
        
        # Sauvegarder le dataset et son état
        torch.save({
            'dataset': dataset,
            'iterator_state': dataset.get_state_dict()
        }, dataset_path)
        print("Dataset initialized and saved.")
    
    return dataset_path

def main():
    # Initialiser le dataset une seule fois
    dataset_path = init_dataset()
    
    world_size = get_gpu_count()
    if world_size == 0:
        print("No GPUs found. Running on CPU")
        subprocess.run([sys.executable, "train.py", "--dataset_path", dataset_path])
        return
        
    print(f"Found {world_size} GPUs. Launching distributed training...")
    
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = "localhost"
    current_env["MASTER_PORT"] = "29500"
    current_env["WORLD_SIZE"] = str(world_size)
    current_env["DATASET_PATH"] = dataset_path
    
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
            for p in processes:
                if p.poll() is None:
                    p.kill()
            sys.exit(1)

if __name__ == "__main__":
    main() 