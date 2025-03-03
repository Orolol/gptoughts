"""
Training script for DeepSeek Mini model.
This version is specifically optimized for training the miniature version of DeepSeek-V3.
"""

import os
import time
import math
import gc
import glob
import threading
import argparse
from queue import Queue
from contextlib import nullcontext
import random

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group, barrier as dist_barrier
from torch.serialization import add_safe_globals

from transformers import AutoTokenizer
from deepseek_mini import DeepSeekMini, DeepSeekMiniConfig
from data.openwebtext.data_loader import StreamingDataset
from run_train import get_datasets

import traceback

from rich.console import Console
console = Console()

import torch._inductor.config
import torch._dynamo.config
from torch.compiler import allow_in_graph

from collections import defaultdict
from contextlib import contextmanager

# Import utility functions
from train_utils import (
    get_gpu_count, setup_distributed, reduce_metrics, calculate_perplexity,
    get_lr, cleanup_old_checkpoints, cleanup_memory, ensure_model_dtype,
    print_memory_stats, setup_cuda_optimizations, save_checkpoint,
    load_checkpoint, find_latest_checkpoint, get_context_manager
)

# Ajouter après les imports
class AveragedTimingStats:
    def __init__(self, print_interval=100):
        self.timings = defaultdict(list)
        self.print_interval = print_interval
        self.current_step = 0
        self.current_context = []
        
    @contextmanager
    def track(self, name):
        self.current_context.append(name)
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            # Only track the first level and its immediate children
            if len(self.current_context) <= 2:
                full_name = '/'.join(self.current_context)
                self.timings[full_name].append(duration)
            self.current_context.pop()
    
    def step(self):
        self.current_step += 1
    
    def should_print(self):
        return self.current_step % self.print_interval == 0
    
    def get_averaged_stats(self):
        if not self.timings:
            return {}
        
        # Calculate averages
        avg_timings = {}
        for name, times in self.timings.items():
            avg_timings[name] = sum(times) / len(times)
        
        # Get total time from root operations
        total_time = sum(avg_timings[name] for name in avg_timings if '/' not in name)
        
        # Calculate percentages
        percentages = {}
        for name, time in avg_timings.items():
            percentages[name] = (time / total_time) * 100 if total_time > 0 else 0
        
        # Clear timings for next window
        self.timings.clear()
        
        return avg_timings, percentages
    
    def print_stats(self):
        timings, percentages = self.get_averaged_stats()
        if not timings:
            return
        
        print(f"\nTiming breakdown over last {self.print_interval} iterations:")
        
        # Sort by time spent (descending)
        sorted_timings = sorted(
            timings.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Print root operations first
        print("\nMain operations:")
        for name, time in sorted_timings:
            if '/' not in name:
                print(f"{name}: {time*1000:.1f}ms ({percentages[name]:.1f}%)")
        
        # Print sub-operations
        print("\nDetailed breakdown:")
        for name, time in sorted_timings:
            if '/' in name:
                print(f"{name}: {time*1000:.1f}ms ({percentages[name]:.1f}%)") 

# Permettre time.time() dans le graphe
allow_in_graph(time.time)

# Configuration optimisée pour les modèles avec dimensions dynamiques
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 16384
# Activer les CUDA graphs même avec des formes dynamiques
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = False
torch._inductor.config.debug = False

# Au début du fichier, après les imports
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"

# -----------------------------------------------------------------------------
# Parse command line arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train DeepSeek Mini model')
parser.add_argument('--size', type=str, choices=['small', 'medium', 'large'], default='small',
                   help='Model size configuration (small, medium, or large)')
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# I/O
out_dir = 'out_deepseek_mini'
eval_interval = 1000
log_interval = 1
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch' or 'resume'

# wandb logging
wandb_log = False
wandb_project = 'deepseek_mini'
wandb_run_name = 'deepseek_mini_run'

# data
dataset = 'openwebtext'
data_dir = 'data/openwebtext'
gradient_accumulation_steps = 1
dropout = 0.0
bias = False
attention_backend = "flash_attn_2" # "sdpa"

# Router loss coefficient
router_z_loss_coef = 0.001  # Weight for the router z-loss

# Configure CUDA Graph behavior
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

# Initialize tokenizer first
access_token = os.getenv('HF_TOKEN')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_fast=True, access_token=access_token)
tokenizer.pad_token = tokenizer.eos_token

# DeepSeek Mini configurations
if torch.cuda.is_available():
    device = f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}'
    
    # Détection automatique de la carte graphique
    gpu_name = torch.cuda.get_device_name()
    is_ampere = any(x in gpu_name.lower() for x in ['a100', 'a6000', 'a5000', 'a4000'])
    is_ada = any(x in gpu_name.lower() for x in ['4090', '4080', '4070'])
    
    if args.size == "small":
        # Configuration de base
        batch_size = 12
        block_size = 512
        
        # Configuration du modèle
        config = DeepSeekMiniConfig(
            vocab_size=len(tokenizer),  # Use tokenizer vocab size
            hidden_size=1024,
            num_hidden_layers=8,
            num_attention_heads=8,
            head_dim=64,
            intermediate_size=2816,
            num_experts=8,
            num_experts_per_token=2,
            max_position_embeddings=block_size,
            kv_compression_dim=64,
            query_compression_dim=192,
            rope_head_dim=32,
            dropout=dropout,
            attention_dropout=dropout,
            hidden_dropout=dropout,
            bias=bias
        )
        
        gradient_accumulation_steps = 2
    
    elif args.size == "medium":
        # Configuration de base
        block_size = 2048
        
        # Configuration du modèle
        config = DeepSeekMiniConfig(
            vocab_size=len(tokenizer),  # Use tokenizer vocab size
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            head_dim=128,
            intermediate_size=4096,
            num_experts=32,
            num_experts_per_token=4,
            max_position_embeddings=block_size,
            kv_compression_dim=128,
            query_compression_dim=384,
            rope_head_dim=32,
            dropout=dropout,
            attention_dropout=dropout,
            hidden_dropout=dropout,
            bias=bias
        )
        
        # Ajustements spécifiques selon le GPU
        if is_ampere:
            batch_size = 48
            gradient_accumulation_steps = 2
            
            # Optimisations mémoire et calcul
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
            
        elif is_ada:
            batch_size = 16
            gradient_accumulation_steps = 4
            torch.set_num_threads(4)
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        else:
            batch_size = 8
            gradient_accumulation_steps = 8
    
    elif args.size == "large":
        # Configuration de base
        block_size = 4096
        
        # Configuration du modèle
        config = DeepSeekMiniConfig(
            vocab_size=len(tokenizer),  # Use tokenizer vocab size
            hidden_size=3072,
            num_hidden_layers=32,
            num_attention_heads=24,
            head_dim=128,
            intermediate_size=8192,
            num_experts=64,
            num_experts_per_token=4,
            max_position_embeddings=block_size,
            kv_compression_dim=256,
            query_compression_dim=768,
            rope_head_dim=32,
            dropout=dropout,
            attention_dropout=dropout,
            hidden_dropout=dropout,
            bias=bias
        )
        
        if is_ampere:
            batch_size = 24
            gradient_accumulation_steps = 4
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif is_ada:
            batch_size = 8
            gradient_accumulation_steps = 8
            torch.set_num_threads(4)
        else:
            batch_size = 4
            gradient_accumulation_steps = 16

    # Apply CUDA optimizations
    setup_cuda_optimizations()
    
    if is_ampere:
        # Optimisations spécifiques A100
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cudnn.benchmark = True
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        torch.cuda.set_per_process_memory_fraction(0.98)
        gc.disable()
    elif is_ada:
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"
        torch.cuda.set_per_process_memory_fraction(0.85)
else:
    device = 'cpu'
    batch_size = 2
    block_size = 64
    config = DeepSeekMiniConfig(
        vocab_size=32000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        head_dim=64,
        intermediate_size=1024,
        num_experts=4,
        num_experts_per_token=2,
        max_position_embeddings=block_size,
        kv_compression_dim=32,
        query_compression_dim=96,
        rope_head_dim=32,
        dropout=dropout,
        attention_dropout=dropout,
        hidden_dropout=dropout,
        bias=bias
    )

# adamw optimizer
learning_rate = 3e-4  # Légèrement plus bas pour la stabilité
max_iters = 600000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95  # Augmenté pour plus de stabilité
grad_clip = 1.0  # Augmenté pour éviter l'explosion des gradients

# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 3e-5

# DDP settings
backend = 'nccl'
compile = False

# Prompts for generation
PROMPT_TEMPLATES = [
    # Science Fiction
    "In the distant future, humanity discovered",
    "The alien civilization revealed",
    # Science
    "The quantum experiment revealed",
    "In the laboratory, scientists discovered",
    "The research paper concluded that",
    # Mathematics
    "The mathematical proof showed",
    "By solving the equation",
    # History
    "During the Industrial Revolution",
    "Ancient civilizations developed",
    # Literature
    "Shakespeare's influence on",
    "The literary analysis demonstrates",
    # Education
    "Modern teaching methods focus on",
    "Students learn best when",
    # Technology
    "Artificial intelligence systems can",
    "The future of computing lies in",
    # Biology
    "The human genome contains",
    "Cellular processes involve",
    # Physics
    "According to quantum mechanics",
    "The laws of thermodynamics state",
    # Chemistry
    "The chemical reaction between",
    "Molecular structures reveal"
]

# -----------------------------------------------------------------------------
# MoE-specific helper functions
# -----------------------------------------------------------------------------

def pad_sequences(encoder_input, decoder_input, target):
    """
    Pad sequences to fixed lengths for better CUDA Graph performance.
    """
    B = encoder_input.size(0)
    
    # Pad encoder input
    if encoder_input.size(1) < encoder_seq_len:
        pad_len = encoder_seq_len - encoder_input.size(1)
        encoder_input = F.pad(encoder_input, (0, pad_len), value=tokenizer.pad_token_id)
    else:
        encoder_input = encoder_input[:, :encoder_seq_len]
    
    # Pad decoder input
    if decoder_input.size(1) < decoder_seq_len:
        pad_len = decoder_seq_len - decoder_input.size(1)
        decoder_input = F.pad(decoder_input, (0, pad_len), value=tokenizer.pad_token_id)
    else:
        decoder_input = decoder_input[:, :decoder_seq_len]
    
    # Pad target
    if target is not None:
        if target.size(1) < decoder_seq_len:
            pad_len = decoder_seq_len - target.size(1)
            target = F.pad(target, (0, pad_len), value=-1)
        else:
            target = target[:, :decoder_seq_len]
    
    return encoder_input, decoder_input, target

def get_batch(data):
    ix = torch.randint(len(data), (batch_size,))
    x = torch.stack([data[i] for i in ix])
    y = torch.stack([data[i] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# Init
# -----------------------------------------------------------------------------

# Initialize distributed training
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    ddp_rank, ddp_local_rank, ddp_world_size, device = setup_distributed(backend=backend)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)

device_type = 'cuda' if 'cuda' in device else 'cpu'
dtype = 'bfloat16' if attention_backend == "flash_attn_2" else 'float16'
print(f"Using dtype: {dtype}")
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = get_context_manager(device_type, dtype)

# Initialize model
if init_from == 'resume':
    print(f"Attempting to resume training from {out_dir}")
    ckpt_path = find_latest_checkpoint(out_dir)
    if not ckpt_path:
        print("No checkpoints found, initializing from scratch instead")
        init_from = 'scratch'
    else:
        print(f"Loading checkpoint: {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Create new model with saved config
        saved_config = DeepSeekMiniConfig(**checkpoint['model_args'])
        model = DeepSeekMini(saved_config)
        
        # Load model and optimizer states
        model, optimizer, iter_num, best_val_loss, _, _ = load_checkpoint(
            ckpt_path, model, map_location='cpu'
        )
        
        if optimizer is None:
            optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
        
        # Move model to device and ensure correct dtype
        model = model.to(device)
        model = ensure_model_dtype(model, ptdtype)
        
        # Reset scaler
        scaler = torch.amp.GradScaler(enabled=(dtype == 'bfloat16' or dtype == 'float16'))
        
        cleanup_memory()

if init_from == 'scratch':
    print("Initializing a new DeepSeek Mini model from scratch")
    model = DeepSeekMini(config)
    model = model.to(device)
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    scaler = torch.amp.GradScaler(enabled=(dtype == 'bfloat16' or dtype == 'float16'))
    iter_num = 1
    best_val_loss = float('inf')

# Print model size
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Initialize timing stats
timing_stats = AveragedTimingStats(print_interval=100)
model.set_timing_stats(timing_stats)

# Initialize DDP if needed
if ddp:
    model = DDP(
        model,
        device_ids=[ddp_local_rank],
        output_device=ddp_local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True
    )
    model.module.set_timing_stats(timing_stats)

# Compile model if requested
if compile:
    print("Compiling model...")
    model.set_gradient_checkpointing(True)
    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"Compilation failed: {e}")
        print(traceback.format_exc())
        compile = False

# Initialize datasets
train_dataset = get_datasets(block_size, batch_size, device, tokenizer=tokenizer)
train_iterator = iter(train_dataset)

# Initialize timing
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0
train_start_time = time.time()
tokens_per_iter = batch_size * block_size * gradient_accumulation_steps
total_tokens = 0
tokens_window = []  # Pour calculer une moyenne glissante des tokens/s
window_size = 10   # Taille de la fenêtre pour la moyenne glissante

print(f"Starting training DeepSeek Mini model")
print(f"Model size: {config.hidden_size}d, {config.num_hidden_layers}l, {config.num_attention_heads}h")
print(f"Batch size: {batch_size}, Block size: {block_size}")
print(f"Using {config.num_experts} experts with top-{config.num_experts_per_token} routing")
print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
print_memory_stats("Initial")

# training loop
while True:
    # determine and set the learning rate for this iteration
    with timing_stats.track("lr_update"):
        lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Generate text every 100 iterations
    if iter_num % 100 == 0 and master_process:
        print("\nText Generation:")
        prompt = random.choice(PROMPT_TEMPLATES)
        
        # Tokenize with proper handling of special tokens
        input_tokens = tokenizer.encode(
            prompt,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_tensors='pt'
        ).to(device)
        
        try:
            with torch.no_grad(), torch.amp.autocast(enabled=True, device_type=device_type):
                # Generate text
                output_ids = model.generate(
                    input_tokens,
                    max_new_tokens=block_size - 10,
                    temperature=0.7,
                    top_k=40
                )
                
                # Decode only the generated part
                generated_text = tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                print(f"Generated text: {prompt} {generated_text}\n")
        except Exception as e:
            print(f"Generation error: {str(e)}")
            print(traceback.format_exc())
            continue

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with gradient accumulation
    with timing_stats.track("optimization"):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0
        total_router_loss = 0
        skip_optimizer_step = False

        try:
            for micro_step in range(gradient_accumulation_steps):
                if ddp:
                    model.require_backward_grad_sync = (
                        micro_step == gradient_accumulation_steps - 1
                    )
                
                # Track data loading time
                with timing_stats.track("data_loading"):
                    batch = next(train_iterator)
                    input_ids = batch['input_ids'].to(device)
                    targets = batch['labels'].to(device)
                
                with timing_stats.track("forward"), torch.amp.autocast(enabled=True, device_type=device_type):
                    # Forward pass with decoder-only architecture
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=None,  # Will be created automatically if needed
                    )
                    
                    # Unpack outputs - model returns a dict
                    hidden_states = outputs["last_hidden_state"]
                    balance_loss = outputs["balance_loss"]
                    
                    # Calculate loss
                    logits = hidden_states @ model.embed_tokens.weight.t() if not ddp else hidden_states @ model.module.embed_tokens.weight.t()
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = targets[..., 1:].contiguous()
                    
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-1
                    )
                    
                    batch_tokens = input_ids.ne(tokenizer.pad_token_id).sum().item()
                    total_tokens += batch_tokens
                    tokens_window.append((time.time(), batch_tokens))
                    if len(tokens_window) > window_size:
                        tokens_window.pop(0)
                    del logits, hidden_states
                
                with timing_stats.track("backward"):
                    if loss is not None:
                        loss = loss / gradient_accumulation_steps
                        balance_loss = balance_loss / gradient_accumulation_steps
                        combined_loss = loss + router_z_loss_coef * balance_loss
                        
                        # Backward pass
                        scaled_loss = scaler.scale(combined_loss)
                        scaled_loss.backward()
                        
                        total_loss += loss.item()
                        total_router_loss += balance_loss.item()
                        del loss, balance_loss, combined_loss, scaled_loss

                with timing_stats.track("optimizer_step"):
                    if grad_clip != 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    
                    scaler.step(optimizer)
                    scaler.update()

        except Exception as e:
            print(f"Training iteration failed: {e}")
            print(traceback.format_exc())
            cleanup_memory()
            if ddp:
                dist_barrier()
            continue

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    if iter_num % log_interval == 0:
        lossf = total_loss
        router_lossf = total_router_loss
        
        # Calculer le taux de tokens/s sur la fenêtre glissante
        if len(tokens_window) > 1:
            window_time = tokens_window[-1][0] - tokens_window[0][0]
            window_tokens = sum(tokens for _, tokens in tokens_window)
            current_tokens_per_sec = window_tokens / window_time if window_time > 0 else 0
        else:
            current_tokens_per_sec = 0
        
        total_time = time.time() - train_start_time
        avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
        
        if local_iter_num >= 5:
            mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        
        print(f"iter {iter_num}: loss {lossf:.4f}, router_loss {router_lossf:.4f}, "
              f"time {dt*1000:.2f}ms, lr {lr:.2e}, "
              f"tt {total_tokens:,}, t/s {current_tokens_per_sec:.2f}, "
              f"avgt/s {avg_tokens_per_sec:.2f}")
        
        timing_stats.step()
        
        if timing_stats.should_print():
            timing_stats.print_stats()

        # Save checkpoint periodically
        if iter_num % 1000 == 0 and master_process:
            save_checkpoint(
                model.module if ddp else model,
                optimizer,
                config.__dict__,  # Use config dict as model args
                iter_num,
                best_val_loss,
                {k: v for k, v in globals().items() if k in config_keys},  # Config
                out_dir
            )
            cleanup_memory()
            cleanup_old_checkpoints(out_dir, keep_num=2)

    iter_num += 1
    local_iter_num += 1
    
    # Periodic memory cleanup
    if iter_num % 100 == 0:
        cleanup_memory()

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

# Final cleanup
cleanup_memory()

if master_process:
    print("Training finished!")

