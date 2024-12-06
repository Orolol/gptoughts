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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT, EncoderDecoderGPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
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
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# Configurations pour l'encoder et le decoder
if torch.cuda.is_available():
    device = 'cuda:0'
    batch_size = 24
    block_size = 64
    
    # Encoder config (plus petit)
    encoder_n_layer = 4
    encoder_n_head = 8
    encoder_n_embd = 768
    encoder_ratio_kv = 8
    
    # Decoder config (plus grand)
    decoder_n_layer = 4
    decoder_n_head = 8
    decoder_n_embd = 768
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

# Chargeur de données modifié pour encoder-decoder
data_dir = os.path.join('data', dataset)
def get_batch(split):
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    
    # Sélectionner des indices aléatoires pour l'encodeur et le décodeur
    encoder_ix = torch.randint(len(data) - block_size, (batch_size,))
    decoder_ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Allouer directement sur GPU
    encoder_x = torch.empty((batch_size, block_size), dtype=torch.long, device=device)
    decoder_x = torch.empty((batch_size, block_size), dtype=torch.long, device=device)
    decoder_y = torch.empty((batch_size, block_size), dtype=torch.long, device=device)
    
    # Remplir par morceaux
    chunk_size = 32
    for i in range(0, batch_size, chunk_size):
        end = min(i + chunk_size, batch_size)
        
        # Données pour l'encodeur
        encoder_chunk_ix = encoder_ix[i:end]
        encoder_data = np.stack([data[i:i+block_size] for i in encoder_chunk_ix])
        encoder_x[i:end] = torch.from_numpy(encoder_data).long().to(device)
        
        # Données pour le décodeur
        decoder_chunk_ix = decoder_ix[i:end]
        decoder_x_data = np.stack([data[i:i+block_size] for i in decoder_chunk_ix])
        decoder_y_data = np.stack([data[i+1:i+1+block_size] for i in decoder_chunk_ix])
        
        decoder_x[i:end] = torch.from_numpy(decoder_x_data).long().to(device)
        decoder_y[i:end] = torch.from_numpy(decoder_y_data).long().to(device)
    
    return encoder_x, decoder_x, decoder_y

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

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            encoder_x, decoder_x, decoder_y = get_batch(split)
            with ctx:
                logits, loss = model(encoder_x, decoder_x, decoder_y)
            losses[k] = loss.item()
        out[split] = losses.mean()
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
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

decode = lambda tokens: tokenizer.decode(tokens, skip_special_tokens=True)

# Liste de débuts de phrases
PROMPT_TEMPLATES = [
    "Once upon a time",
    "In a distant future",
    "The king said",
    "She looked at him and",
    "Deep in the forest",
    "As the sun rose",
    "The old wizard",
    "In the castle",
    "The dragon",
    "By the river"
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

# training loop
encoder_x, decoder_x, decoder_y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Génération de texte toutes les 100 itérations
    if iter_num % 100 == 0 and master_process:
        print(f"\n--- Génération d'exemple à l'itération {iter_num} ---")
        generated_tokens = generate_text(raw_model, encoder_x[0])
        print("Texte généré:", decode(generated_tokens))
        print("--------------------\n")

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # Prefetch le prochain batch
    with torch.cuda.stream(copy_stream):
        encoder_x_next, decoder_x_next, decoder_y_next = get_batch('train')
    
    # S'assurer que le batch précédent est terminé
    default_stream.wait_stream(copy_stream)
    
    # Forward pass avec mixed precision
    with ctx:
        logits, loss = model(encoder_x, decoder_x, decoder_y)
        loss = loss / gradient_accumulation_steps
    
    # Backward pass optimisé
    scaler.scale(loss).backward()
    
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    
    # Swap les batches
    encoder_x, decoder_x, decoder_y = encoder_x_next, decoder_x_next, decoder_y_next

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

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
