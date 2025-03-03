"""
Training script for DeepSeek model.
This version is optimized for training the DeepSeek model with MoE and MLA attention.
"""

import os
import time
import math
import gc
import glob
import random
import argparse
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from deepseek import Transformer, ModelArgs

import torch._inductor.config
import torch._dynamo.config
from torch.compiler import allow_in_graph

from collections import defaultdict
from contextlib import contextmanager

from data_loader import FinewebDataset

# Allow time.time() in graph
allow_in_graph(time.time)

# Optimized configuration for models with dynamic dimensions
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 16384
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = False
torch._inductor.config.debug = False

# CUDA memory configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"

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
        
        avg_timings = {}
        for name, times in self.timings.items():
            avg_timings[name] = sum(times) / len(times)
        
        total_time = sum(avg_timings[name] for name in avg_timings if '/' not in name)
        
        percentages = {}
        for name, time in avg_timings.items():
            percentages[name] = (time / total_time) * 100 if total_time > 0 else 0
        
        self.timings.clear()
        
        return avg_timings, percentages
    
    def print_stats(self):
        timings, percentages = self.get_averaged_stats()
        if not timings:
            return
        
        print(f"\nTiming breakdown over last {self.print_interval} iterations:")
        
        sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)
        
        print("\nMain operations:")
        for name, time in sorted_timings:
            if '/' not in name:
                print(f"{name}: {time*1000:.1f}ms ({percentages[name]:.1f}%)")
        
        print("\nDetailed breakdown:")
        for name, time in sorted_timings:
            if '/' in name:
                print(f"{name}: {time*1000:.1f}ms ({percentages[name]:.1f}%)")

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.synchronize()

def print_memory_stats(prefix=""):
    if torch.cuda.is_available():
        print(f"\n{prefix} Memory stats:")
        print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
        print(f"Max reserved: {torch.cuda.max_memory_reserved()/1e9:.2f}GB")
        torch.cuda.reset_peak_memory_stats()

def get_batch(data_loader):
    try:
        batch = next(data_loader)
        # Ensure all tensors are contiguous
        return {k: v.contiguous() if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
    except StopIteration:
        # Reset iterator
        data_loader = iter(data_loader.dataset)
        batch = next(data_loader)
        return {k: v.contiguous() if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data[split])
            with ctx:
                logits = model(X)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# I/O
out_dir = 'out_deepseek'
eval_interval = 1000
log_interval = 1
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch' or 'resume'

# wandb logging
wandb_log = False
wandb_project = 'deepseek'
wandb_run_name = 'deepseek_run'

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 1
dropout = 0.0
bias = False

# adamw optimizer
learning_rate = 3e-4
max_iters = 600000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 3e-5

# DDP settings
backend = 'nccl'
compile = False

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() else 'float32'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Initialize
# -----------------------------------------------------------------------------

# Initialize distributed training
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)

# Initialize tokenizer first (before model configuration)
access_token = os.getenv('HF_TOKEN')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_fast=True, access_token=access_token)
tokenizer.pad_token = tokenizer.eos_token

# model configuration
block_size = 64
n_layer = 8
n_head = 8
n_embd = 1024
multiple_of = 256
dropout = 0.0

# Initialize model with more specific arguments
model_args = ModelArgs(
    max_batch_size=128,  # Increased to accommodate flattened batch size (8*10=80)
    max_seq_len=block_size,
    vocab_size=len(tokenizer),
    dim=n_embd,  # 1024
    inter_dim=5472,  # Scaled down from 10944 proportionally
    moe_inter_dim=704,  # Scaled down from 1408 proportionally
    n_layers=n_layer,  # 16 layers
    n_dense_layers=1,  # Keep the same
    n_heads=n_head,  # 16 heads
    n_routed_experts=4,  # Reduced from 64
    n_shared_experts=2,  # Keep the same
    n_activated_experts=2,  # Reduced from 6
    score_func="softmax",  # Default value from ModelArgs
    route_scale=1.0,  # Keep the same
    q_lora_rank=0,  # Keep the same
    kv_lora_rank=256,  # Reduced from 512
    qk_nope_head_dim=64,  # Reduced from 128
    qk_rope_head_dim=32,  # Reduced from 64
    v_head_dim=64,  # Reduced from 128
    original_seq_len=4096,  # Default value from ModelArgs
    rope_theta=10000.0,  # Default value from ModelArgs
    rope_factor=40,  # Default value from ModelArgs
    beta_fast=32,  # Default value from ModelArgs
    beta_slow=1,  # Default value from ModelArgs
    mscale=0.707  # Keep the same
)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    model = Transformer(model_args)
    model.to(device)
else:
    print(f"Resuming training from {out_dir}")
    # Load the checkpoint
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # Create the model
    model = Transformer(checkpoint_model_args)
    state_dict = checkpoint['model']
    # Fix the keys of the state dictionary
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    model.to(device)

# Initialize optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    weight_decay=weight_decay
)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# Initialize grad scaler
scaler = torch.amp.GradScaler(enabled=(dtype == 'bfloat16' or dtype == 'float16'))

# Initialize DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Initialize timing stats
timing_stats = AveragedTimingStats(print_interval=100)

# Compile model if requested
if compile:
    print("Compiling model...")
    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"Compilation failed: {e}")
        compile = False

# Initialize data
if ddp:
    # Calculate offset for each process in DDP
    per_worker_batch_size = 8 // ddp_world_size
    start_offset = ddp_rank * per_worker_batch_size
else:
    per_worker_batch_size = 8
    start_offset = 0

train_dataset = FinewebDataset(
    split='train',
    max_length=block_size,
    buffer_size=10,
    shuffle=True,
    start_offset=start_offset,
    tokenizer=tokenizer
)

train_data = DataLoader(
    dataset=train_dataset,
    batch_size=per_worker_batch_size,
    num_workers=4,
    pin_memory=True
)

# Initialize training state
iter_num = 0 if init_from == 'scratch' else checkpoint['iter_num']
best_val_loss = float('inf') if init_from == 'scratch' else checkpoint['best_val_loss']
train_iter = iter(train_data)

# Training loop
print(f"Starting training")
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model

while True:
    # Determine learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Get batch
    with timing_stats.track("data_loading"):
        print(f"train_iter: {train_iter}")
        batch = get_batch(train_iter)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

    # Forward backward update
    with timing_stats.track("optimization"):
        print(f"Optim")

        with ctx:
            # Forward pass with attention mask
            seqlen = input_ids.size(1) if len(input_ids.shape) == 2 else input_ids.size(2)
            # Create attention mask for causal attention
            mask = None
            if seqlen > 1:
                mask = torch.full((seqlen, seqlen), float("-inf"), device=input_ids.device).triu_(1)
            
            print(f"input_ids: {input_ids.shape}")
            # Forward pass with tokens
            logits = model(tokens=input_ids, start_pos=0)
            
            # Reshape tensors for loss calculation
            if len(input_ids.shape) == 3:
                batch_size, buffer_size, seq_len = input_ids.shape
                # Reshape labels to match logits shape before flattening
                labels = labels.reshape(-1, seq_len)
            
            # Compute loss directly on logits
            # For causal language modeling, we predict the next token 
            # (shift logits left and targets right by 1)
            if logits.dim() > 2:
                # If logits are [batch, seq, vocab]
                logits = logits.view(-1, logits.size(-1))  # [batch*seq, vocab]
                
            loss = F.cross_entropy(
                logits,  # Already reshaped to [batch*seq, vocab]
                labels.view(-1),
                ignore_index=-1
            )
            
        print(f"Loss: {loss}")
        
        # Scale loss and backward pass
        scaler.scale(loss).backward()
        
        print(f"Scaler: {scaler}")
        # Clip gradients
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        print(f"Optimizer: {optimizer}")
        # Step optimizer and scaler
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        print(f"Optimizer: {optimizer}")

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    if iter_num % log_interval == 0 and master_process:
        print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms, lr {lr:e}")
        timing_stats.step()
        if timing_stats.should_print():
            timing_stats.print_stats()

    # Evaluation
    if iter_num > 0 and iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Save checkpoint
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args.__dict__,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    iter_num += 1
    local_iter_num += 1

    # End training at max_iters
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group() 