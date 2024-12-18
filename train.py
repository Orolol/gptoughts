"""
Training script for GPT model with distributed data parallel support.
Includes both single GPU and multi-GPU (DDP) training capabilities.
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import threading
from queue import Queue
import gc
import glob
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import AutoTokenizer

from model import GPTConfig, GPT, EncoderDecoderGPT
from data.openwebtext.data_loader import StreamingDataset

# Configuration
class TrainingConfig:
    def __init__(self):
        # I/O settings
        self.out_dir = 'out'
        self.eval_interval = 100000
        self.log_interval = 1
        self.generate_interval = 20
        self.eval_iters = 100
        self.eval_only = False
        self.always_save_checkpoint = True
        self.init_from = 'scratch'
        
        # Wandb settings
        self.wandb_log = False
        self.wandb_project = 'owt'
        self.wandb_run_name = 'gpt2'
        
        # Training settings
        self.dataset = 'openwebtext'
        self.data_dir = 'data/openwebtext'
        self.dropout = 0.0
        self.bias = False
        
        # Optimizer settings
        self.learning_rate = 3e-4
        self.max_iters = 600000
        self.weight_decay = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0
        
        # LR decay settings
        self.decay_lr = True
        self.warmup_iters = 2000
        self.lr_decay_iters = 600000
        self.min_lr = 6e-5
        
        # DDP settings
        self.backend = 'nccl'
        self.compile = False
        
        # Device specific settings
        self.setup_device_specific_config()
        
    def setup_device_specific_config(self):
        if not torch.cuda.is_available():
            self.setup_cpu_config()
            return

        # Détecter le type de GPU
        gpu_name = torch.cuda.get_device_name(0).lower()
        
        if 'h100' in gpu_name:
            self.setup_h100_config()
        elif '4090' in gpu_name or '40' in gpu_name:
            self.setup_4090_config()
        elif 'a100' in gpu_name:
            self.setup_a100_config()
        elif '2070' in gpu_name:
            self.setup_2070_super_config()
        else:
            # Configuration par défaut pour les autres GPUs
            self.setup_default_gpu_config()
    
    def setup_h100_config(self):
        """Configuration optimisée pour H100"""
        self.device = 'cuda:0'
        self.batch_size = 16  # Plus grand batch size grâce à la grande mémoire
        self.block_size = 512
        self.encoder_config = {
            'n_layer': 16,
            'n_head': 16,
            'n_embd': 768,
            'ratio_kv': 8
        }
        self.decoder_config = {
            'n_layer': 16,
            'n_head': 16,
            'n_embd': 768,
            'ratio_kv': 8
        }
        self.gradient_accumulation_steps = 1
        self.dtype = 'bfloat16'  # H100 optimisé pour bfloat16
        
        # Optimisations spécifiques H100
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(0.95)
        
    def setup_4090_config(self):
        """Configuration optimisée pour RTX 4090"""
        self.device = 'cuda:0'
        self.batch_size = 8
        self.block_size = 256
        self.encoder_config = {
            'n_layer': 16,
            'n_head': 16,
            'n_embd': 768,
            'ratio_kv': 8
        }
        self.decoder_config = {
            'n_layer': 16,
            'n_head': 16,
            'n_embd': 768,
            'ratio_kv': 8
        }
        self.gradient_accumulation_steps = 2
        self.dtype = 'float16'  # 4090 plus efficace avec float16
        
        # Optimisations spécifiques 4090
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(0.90)
        
    def setup_a100_config(self):
        """Configuration optimisée pour A100"""
        self.device = 'cuda:0'
        self.batch_size = 24
        self.block_size = 1536
        self.encoder_config = {
            'n_layer': 28,
            'n_head': 36,
            'n_embd': 1536,
            'ratio_kv': 8
        }
        self.decoder_config = {
            'n_layer': 28,
            'n_head': 36,
            'n_embd': 1536,
            'ratio_kv': 8
        }
        self.gradient_accumulation_steps = 1
        self.dtype = 'bfloat16'  # A100 optimisé pour bfloat16
        
        # Optimisations spécifiques A100
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(0.90)
        
    def setup_2070_super_config(self):
        """Configuration optimisée pour RTX 2070 Super"""
        self.device = 'cuda:0'
        self.batch_size = 16  # Batch size réduit pour la mémoire limitée
        self.block_size = 64
        self.encoder_config = {
            'n_layer': 4,
            'n_head': 4,
            'n_embd': 768,
            'ratio_kv': 4
        }
        self.decoder_config = {
            'n_layer': 8,
            'n_head': 8,
            'n_embd': 768,
            'ratio_kv': 4
        }
        self.gradient_accumulation_steps = 8  # Augmenté pour compenser le petit batch size
        self.dtype = 'float16'  # float16 pour les GPU RTX 20xx
        
        # Optimisations spécifiques 2070 Super
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(0.85)  # Garde une marge pour éviter les OOM
        
        # Optimisations supplémentaires pour la série 20xx
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.cuda, 'memory_stats'):
            torch.cuda.memory_stats()
            
    def setup_default_gpu_config(self):
        """Configuration par défaut pour les autres GPUs"""
        self.device = 'cuda:0'
        self.batch_size = 8
        self.block_size = 512
        self.encoder_config = {
            'n_layer': 12,
            'n_head': 16,
            'n_embd': 768,
            'ratio_kv': 4
        }
        self.decoder_config = {
            'n_layer': 12,
            'n_head': 16,
            'n_embd': 768,
            'ratio_kv': 4
        }
        self.gradient_accumulation_steps = 4
        self.dtype = 'float16'
        
        # Optimisations de base
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(0.80)
        
    def setup_cpu_config(self):
        self.device = 'cpu'
        self.batch_size = 2
        self.block_size = 64
        self.encoder_config = {
            'n_layer': 1,
            'n_head': 1,
            'n_embd': 2,
            'ratio_kv': 1
        }
        self.decoder_config = {
            'n_layer': 1,
            'n_head': 1,
            'n_embd': 2,
            'ratio_kv': 1
        }
        self.dtype = 'float32'

class TextGenerator:
    PROMPT_TEMPLATES = [
        "Il était une fois",
        "Dans un futur lointain",
        "Le roi dit",
        "Elle le regarda et",
        "Au fond de la forêt",
        "Quand le soleil se leva",
        "Le vieux sorcier",
        "Dans le château",
        "Le dragon",
        "Au bord de la rivière"
    ]
    
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        
    @torch.no_grad()
    def generate(self, model, max_new_tokens=50, temperature=0.8):
        model.eval()
        prompt = random.choice(self.PROMPT_TEMPLATES)
        prompt_tokens = torch.tensor(
            self.tokenizer.encode(prompt, add_special_tokens=False),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)
        
        dtype = model.dtype if hasattr(model, 'dtype') else next(model.parameters()).dtype
        with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            result = model.generate(prompt_tokens, max_new_tokens=max_new_tokens, temperature=temperature)
            # Now we print the generated text
            print(prompt + " " + self.tokenizer.decode(result[0].tolist()))
            return result

class Trainer:
    def __init__(self, config):
        self.config = config
        self.setup_environment()
        self.setup_tokenizer()
        self.setup_model()
        self.setup_optimizer()
        self.text_generator = TextGenerator(self.tokenizer, self.device)
        
    def setup_environment(self):
        # Setup DDP if needed
        self.setup_distributed_training()
        
        # Setup device and seeds
        torch.manual_seed(1337 + self.seed_offset)
        self.device_type = 'cuda' if 'cuda' in self.config.device else 'cpu'
        self.ptdtype = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16
        }[self.config.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)
        
    def setup_distributed_training(self):
        self.ddp = int(os.environ.get('RANK', -1)) != -1
        if self.ddp:
            init_process_group(backend=self.config.backend)
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.ddp_local_rank}'
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0
            self.seed_offset = self.ddp_rank
        else:
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1
            self.device = self.config.device
            
    def setup_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            use_fast=True,
            access_token=os.getenv('HF_TOKEN')
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def setup_model(self):
        # Model initialization logic here
        vocab_size = len(self.tokenizer)
        encoder_config = GPTConfig(
            n_layer=self.config.encoder_config['n_layer'],
            n_head=self.config.encoder_config['n_head'],
            n_embd=self.config.encoder_config['n_embd'],
            block_size=self.config.block_size,
            ratio_kv=self.config.encoder_config['ratio_kv'],
            bias=self.config.bias,
            vocab_size=vocab_size,
            dropout=self.config.dropout
        )
        decoder_config = GPTConfig(
            n_layer=self.config.decoder_config['n_layer'],
            n_head=self.config.decoder_config['n_head'],
            n_embd=self.config.decoder_config['n_embd'],
            block_size=self.config.block_size,
            ratio_kv=self.config.decoder_config['ratio_kv'],
            bias=self.config.bias,
            vocab_size=vocab_size,
            dropout=self.config.dropout
        )
        self.model = EncoderDecoderGPT(encoder_config, decoder_config)
        self.model.to(self.device)
        
        # Wrap model in DDP if using distributed training
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])
        
    def setup_optimizer(self):
        # Get the underlying model if using DDP
        model = self.model.module if self.ddp else self.model
        # Optimizer initialization logic here
        self.optimizer = model.configure_optimizers(
            self.config.weight_decay,
            self.config.learning_rate,
            (self.config.beta1, self.config.beta2),
            self.device_type
        )
        # Initialize gradient scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()
        
    def train(self):
        # Training loop logic here
        
        print("Loading datasets...")
        train_dataset = StreamingDataset(
            block_size=self.config.block_size,
            batch_size=self.config.batch_size,
            dataset_name="HuggingFaceFW/fineweb-2",
            dataset_config="fra_Latn",
            split="train",
            device=self.device
        )
        val_dataset = StreamingDataset(
            block_size=self.config.block_size,
            batch_size=self.config.batch_size,
            dataset_name="HuggingFaceFW/fineweb-2",
            dataset_config="fra_Latn",
            split="test",
            device=self.device
        )
        
        iter_num = 1
        best_val_loss = 1e9
        print("Training...")
        while True:
            # Determine and set the learning rate for this iteration
            lr = self.get_lr(iter_num) if self.config.decay_lr else self.config.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Evaluate the loss on train/val sets and write checkpoints
            if iter_num % self.config.eval_interval == 0 and self.master_process and iter_num > 0:
                losses = self.estimate_loss(train_dataset, val_dataset)
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if losses['val'] < best_val_loss or self.config.always_save_checkpoint:
                    best_val_loss = losses['val']
                    self.save_checkpoint(iter_num, best_val_loss, losses['val'])
            
            
            
            # Forward backward update, with optional gradient accumulation
            encoder_input, target = next(iter(train_dataset))
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                logits, loss = self.model(encoder_input, encoder_input, target)
                # we should print loss, time  per step and total number of tokens processed
                print(f"")
                print(f"step {iter_num}: train loss {loss.item():.4f}")
                if loss is not None:
                    loss = loss / self.config.gradient_accumulation_steps
                else:
                    continue
            
            # Scaler pour mixed precision
            self.scaler.scale(loss).backward()
            
            if self.config.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if iter_num % self.config.generate_interval == 0 and self.master_process:
                self.text_generator.generate(self.model)
            
            iter_num += 1
            
            # Termination conditions
            if iter_num > self.config.max_iters:
                break
        
    def get_lr(self, it):
        if it < self.config.warmup_iters:
            return self.config.learning_rate * it / self.config.warmup_iters
        if it > self.config.lr_decay_iters:
            return self.config.min_lr
        decay_ratio = (it - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)
    
    def estimate_loss(self, train_dataset, val_dataset):
        out = {}
        self.model.eval()
        for split, dataset in [('train', train_dataset), ('val', val_dataset)]:
            losses = torch.zeros(self.config.eval_iters)
            valid_iters = 0
            for k in range(self.config.eval_iters):
                encoder_input, target = next(iter(dataset))
                with self.ctx:
                    logits, loss = self.model(encoder_input, encoder_input, target)
                    if loss is not None:
                        losses[valid_iters] = loss.item()
                        valid_iters += 1
            
            # Average only over valid iterations
            out[split] = losses[:valid_iters].mean() if valid_iters > 0 else float('inf')
        self.model.train()
        return out
    
    def save_checkpoint(self, iter_num, best_val_loss, val_loss):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'config': self.config,
        }
        checkpoint_path = os.path.join(
            self.config.out_dir, 
            f'ckpt_iter_{iter_num}_loss_{val_loss:.4f}.pt'
        )
        print(f"saving checkpoint to {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)
        
        # Nettoyer les anciens checkpoints
        self.cleanup_old_checkpoints(self.config.out_dir, keep_last_n=4)
    
    def cleanup_old_checkpoints(self, out_dir, keep_last_n=4):
        # Ne garde que les N derniers checkpoints
        checkpoints = glob.glob(os.path.join(out_dir, 'ckpt_iter_*.pt'))
        if len(checkpoints) <= keep_last_n:
            return
        
        def extract_iter_num(filename):
            try:
                return int(filename.split('iter_')[1].split('_loss')[0])
            except:
                return 0
        
        # Trier les checkpoints par numéro d'itération
        checkpoints.sort(key=extract_iter_num)
        
        # Supprimer les plus anciens
        for ckpt in checkpoints[:-keep_last_n]:
            try:
                os.remove(ckpt)
                print(f"Removed old checkpoint: {ckpt}")
            except Exception as e:
                print(f"Error removing checkpoint {ckpt}: {e}")

def main():
    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        # Launch with torch.distributed.launch
        import sys
        if not int(os.environ.get('RANK', -1)) >= 0:
            print(f"Detected {num_gpus} GPUs. Launching distributed training...")
            cmd = [sys.executable, "-m", "torch.distributed.launch",
                  f"--nproc_per_node={num_gpus}",
                  "--use_env",
                  __file__] + sys.argv[1:]
            import subprocess
            subprocess.run(cmd)
            sys.exit()
    
    config = TrainingConfig()
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
