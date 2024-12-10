"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT, EncoderDecoderGPT
from data.openwebtext.data_loader import StreamingDataset

access_token=os.getenv('HF_TOKEN')

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
generate_interval = 50
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
data_dir = 'data/openwebtext'
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# Configurations pour l'encoder et le decoder
if torch.cuda.is_available():
    device = 'cuda:0'
    batch_size = 32
    block_size = 64
    
    # Encoder config (plus petit)
    encoder_n_layer = 4
    encoder_n_head = 8
    encoder_n_embd = 512
    encoder_ratio_kv = 8
    
    # Decoder config (plus grand)
    decoder_n_layer = 12
    decoder_n_head = 12
    decoder_n_embd = 512
    decoder_ratio_kv = 8
    
    # Optimisations mémoire
    gradient_accumulation_steps = 1
    dtype = 'bfloat16'
    
    # Activer la gestion de mémoire optimisée
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Activer la mémoire partagée
    if hasattr(torch.cuda, 'memory_stats'):
        torch.cuda.memory_stats()
    torch.cuda.set_device(device)
else:
    device = 'cpu'
    batch_size = 16
    block_size = 8
    
    # Encoder config (minimal)
    encoder_n_layer = 1
    encoder_n_head = 1
    encoder_n_embd = 8
    encoder_ratio_kv = 1
    
    # Decoder config (minimal)
    decoder_n_layer = 1
    decoder_n_head = 1
    decoder_n_embd = 8
    decoder_ratio_kv = 1

# adamw optimizer
learning_rate = 3e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Charger le décodeur de tokens
from transformers import AutoTokenizer
import random
import pickle
import os

# Charger le tokenizer depuis meta.pkl
meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    print("Loading tokenizer from meta.pkl...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    tokenizer = meta['tokenizer']
else:
    print("meta.pkl not found, loading default tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_fast=True, access_token=access_token)
    tokenizer.pad_token = tokenizer.eos_token

decode = lambda tokens: tokenizer.decode(tokens, skip_special_tokens=True)

# Liste de débuts de phrases
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

@torch.no_grad()
def generate_text(model, encoder_input, max_new_tokens=50, temperature=0.8):
    model.eval()
    
    # Choisir un début de phrase aléatoire et l'encoder
    prompt = random.choice(PROMPT_TEMPLATES)
    prompt_tokens = torch.tensor(
        tokenizer.encode(prompt, add_special_tokens=False), 
        dtype=torch.long, 
        device=encoder_input.device
    )
    
    # Encoder l'entrée avec le prompt
    encoder_output = encoder_input.unsqueeze(0)
    
    # Utiliser le prompt comme entrée du décodeur
    decoder_input = prompt_tokens.unsqueeze(0)
    
    # Générer token par token
    generated_tokens = model.generate(encoder_output, decoder_input, max_new_tokens=max_new_tokens, temperature=temperature)
    
    model.train()
    return generated_tokens[0].tolist()

# init these up here, can override if init_from='resume'
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304

# Créer les configurations pour l'encodeur et le décodeur
encoder_config = GPTConfig(
    n_layer=encoder_n_layer,
    n_head=encoder_n_head,
    n_embd=encoder_n_embd,
    block_size=block_size,
    ratio_kv=encoder_ratio_kv,
    bias=bias,
    vocab_size=vocab_size,
    dropout=dropout
)

decoder_config = GPTConfig(
    n_layer=decoder_n_layer,
    n_head=decoder_n_head,
    n_embd=decoder_n_embd,
    block_size=block_size,
    ratio_kv=decoder_ratio_kv,
    bias=bias,
    vocab_size=vocab_size,
    dropout=dropout
)

# Store model arguments for checkpointing
model_args = {
    'encoder_config': encoder_config,
    'decoder_config': decoder_config,
    'vocab_size': vocab_size,
    'block_size': block_size
}

if init_from == 'scratch':
    print("Initializing a new encoder-decoder model from scratch")
    model = EncoderDecoderGPT(encoder_config, decoder_config)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = EncoderDecoderGPT(encoder_config, decoder_config)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

model.to(device)

# Optimisations CUDA
if device_type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    
    default_stream = torch.cuda.current_stream()
    copy_stream = torch.cuda.Stream()
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    device_index = 0 if isinstance(device, str) else device
    if isinstance(device, str) and ':' in device:
        device_index = int(device.split(':')[1])
    torch.cuda.set_device(device_index)
    torch.cuda.empty_cache()
    
    pin_memory = True

# Configurer le gradient scaler
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'bfloat16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Initialize datasets
train_dataset = StreamingDataset(
    block_size=block_size,
    batch_size=batch_size,
    dataset_name="HuggingFaceFW/fineweb-2",
    dataset_config="fra_Latn",
    split="train",
    device=device
)

val_dataset = StreamingDataset(
    block_size=block_size,
    batch_size=batch_size,
    dataset_name="HuggingFaceFW/fineweb-2",
    dataset_config="fra_Latn",
    split="test",
    device=device
)

# Get vocab size from the dataset
vocab_size = len(train_dataset.tokenizer)
print(f"vocab_size: {vocab_size}")

# Update the estimate_loss function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, dataset in [('train', train_dataset), ('val', val_dataset)]:
        losses = torch.zeros(eval_iters)
        valid_iters = 0
        for k in range(eval_iters):
            encoder_input, decoder_input, target = next(iter(dataset))
            with ctx:
                logits, loss = model(encoder_input, decoder_input, target)
                if loss is not None:
                    losses[valid_iters] = loss.item()
                    valid_iters += 1
        
        # Average only over valid iterations
        out[split] = losses[:valid_iters].mean() if valid_iters > 0 else float('inf')
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

print("Initializing training loop...")

# training loop
train_iterator = iter(train_dataset)
encoder_input, decoder_input, target = next(train_iterator)
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

print("Starting training...")

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process and iter_num > 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(f"Total tokens processed: {train_dataset.get_total_tokens():,}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
                "total_tokens": train_dataset.get_total_tokens()
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(encoder_input, decoder_input, target)
            if loss is not None:
                loss = loss / gradient_accumulation_steps
            else:
                continue
            
            # immediately async prefetch next batch
            try:
                encoder_input_next, decoder_input_next, target_next = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataset)
                encoder_input_next, decoder_input_next, target_next = next(train_iterator)
            
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    if iter_num % generate_interval == 0 and master_process:
        print(generate_text(model, encoder_input, max_new_tokens=50, temperature=0.8))
    iter_num += 1
    local_iter_num += 1
    encoder_input, decoder_input, target = encoder_input_next, decoder_input_next, target_next

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

# Nettoyage final CUDA
if device_type == 'cuda':
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, 'memory_stats'):
        torch.cuda.memory_stats(device=device)
    torch.cuda.set_stream(torch.cuda.Stream())
