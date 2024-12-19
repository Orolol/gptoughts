"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).
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

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from rich.console import Console
from rich.progress import Progress
from rich import print as rprint

from model import GPTConfig, GPT, EncoderDecoderGPT
from data.openwebtext.data_loader import StreamingDataset
from config import TrainingConfig

# Initialize rich console for pretty printing
console = Console()

# Load configuration
training_config = TrainingConfig()

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = training_config.out_dir
eval_interval = training_config.eval_interval
log_interval = training_config.log_interval
generate_interval = training_config.generate_interval
eval_iters = training_config.eval_iters
eval_only = training_config.eval_only # if True, script exits right after the first eval
always_save_checkpoint = training_config.always_save_checkpoint # if True, always save a checkpoint after each eval
init_from = training_config.init_from # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = training_config.wandb_log # disabled by default
wandb_project = training_config.wandb_project
wandb_run_name = training_config.wandb_run_name # 'run' + str(time.time())
# data
dataset = training_config.dataset
data_dir = training_config.data_dir
gradient_accumulation_steps = training_config.gradient_accumulation_steps # used to simulate larger batch sizes
dropout = training_config.dropout # for pretraining 0 is good, for finetuning try 0.1+
bias = training_config.bias # do we use bias inside LayerNorm and Linear layers?

# Configurations pour l'encoder et le decoder
if torch.cuda.is_available():
    device = 'cuda:0'
    batch_size = training_config.batch_size
    block_size = training_config.block_size
    
    # Encoder config (plus petit)
    encoder_n_layer = training_config.encoder_n_layer
    encoder_n_head = training_config.encoder_n_head
    encoder_n_embd = training_config.encoder_n_embd
    encoder_ratio_kv = training_config.encoder_ratio_kv
    
    # Decoder config (plus grand)
    decoder_n_layer = training_config.decoder_n_layer
    decoder_n_head = training_config.decoder_n_head
    decoder_n_embd = training_config.decoder_n_embd
    decoder_ratio_kv = training_config.decoder_ratio_kv
    
    # Optimisations mémoire
    gradient_accumulation_steps = training_config.gradient_accumulation_steps
    dtype = training_config.dtype
    
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
    batch_size = training_config.batch_size
    block_size = training_config.block_size
    
    # Encoder config (minimal)
    encoder_n_layer = training_config.encoder_n_layer
    encoder_n_head = training_config.encoder_n_head
    encoder_n_embd = training_config.encoder_n_embd
    encoder_ratio_kv = training_config.encoder_ratio_kv
    
    # Decoder config (minimal)
    decoder_n_layer = training_config.decoder_n_layer
    decoder_n_head = training_config.decoder_n_head
    decoder_n_embd = training_config.decoder_n_embd
    decoder_ratio_kv = training_config.decoder_ratio_kv

# adamw optimizer
learning_rate = training_config.learning_rate # max learning rate
max_iters = training_config.max_iters # total number of training iterations
weight_decay = training_config.weight_decay
beta1 = training_config.beta1
beta2 = training_config.beta2
grad_clip = training_config.grad_clip # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = training_config.decay_lr # whether to decay the learning rate
warmup_iters = training_config.warmup_iters # how many steps to warm up for
lr_decay_iters = training_config.lr_decay_iters # should be ~= max_iters per Chinchilla
min_lr = training_config.min_lr # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = training_config.backend # 'nccl', 'gloo', etc.

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = training_config.compile # use PyTorch 2.0 to compile the model to be faster
# ----------------------------------------------------------------------------------------------------------------------------------------------

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

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_fast=True, access_token=training_config.access_token)
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
def generate_text(model, input, max_new_tokens=50, temperature=0.8):
    model.eval()
    
    # Choisir un début de phrase aléatoire et l'encoder
    prompt = random.choice(PROMPT_TEMPLATES)
    prompt_tokens = torch.tensor(
        tokenizer.encode(prompt, add_special_tokens=False), 
        dtype=torch.long, 
        device=input.device
    )
    
    # Ajouter la dimension de batch
    prompt_tokens = prompt_tokens.unsqueeze(0)  # [1, seq_len]
    
    # Convertir au même dtype que le modèle
    if hasattr(model, 'dtype'):
        dtype = model.dtype
    else:
        # Détecter le dtype à partir des paramètres du modèle
        dtype = next(model.parameters()).dtype
        
    # S'assurer que le modèle et les tenseurs sont dans le même dtype
    with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
        # Générer token par token
        generated_tokens = model.generate(prompt_tokens, max_new_tokens=max_new_tokens, temperature=temperature)
    
    # Décoder les tokens générés
    decoded_text = decode(generated_tokens[0].tolist())
    
    model.train()
    return prompt + " " + decoded_text  # Retourner directement le texte décodé

# init these up here, can override if init_from='resume'
iter_num = 0
best_val_loss = 1e9


# model init
vocab_size = len(tokenizer)

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

if init_from == 'resume':
    print(f"Attempting to resume training from {out_dir}")
    # Trouver tous les checkpoints
    checkpoints = glob.glob(os.path.join(out_dir, 'ckpt_iter_*.pt'))
    if not checkpoints:
        print("No checkpoints found, initializing from scratch instead")
        init_from = 'scratch'
        model = EncoderDecoderGPT(encoder_config, decoder_config)
    else:
        # Extraire les numéros d'itération et trier
        def extract_iter_num(filename):
            """Extrait le numéro d'itération d'un nom de fichier checkpoint."""
            try:
                return int(filename.split('ckpt_')[1].split('.pt')[0])
            except:
                return 0
        
        # Prendre le dernier checkpoint
        checkpoints.sort(key=extract_iter_num)
        ckpt_path = checkpoints[-1]
        print(f"Loading checkpoint: {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        model = EncoderDecoderGPT(encoder_config, decoder_config)
        state_dict = checkpoint['model']
        
        # Nettoyer le state dict si nécessaire
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming from iteration {iter_num} with best val loss {best_val_loss}")

if init_from == 'scratch':
    print("Initializing a new encoder-decoder model from scratch")
    model = EncoderDecoderGPT(encoder_config, decoder_config)

model.to(device)

# Wrap model in DDP if using distributed training
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Pour sauvegarder le modèle, nous avons besoin d'accéder au modèle sous-jacent
raw_model = model.module if ddp else model

# Initialize optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

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
    
    # Désactiver le garbage collector automatique
    gc.disable()
    
    # Configurer la gestion de la mémoire CUDA
    torch.cuda.empty_cache()
    torch.cuda.memory.empty_cache()
    torch.backends.cudnn.benchmark = True
    
    # Réserver de la mémoire CUDA
    cache_size = 24 * 1024 * 1024 * 1024  # 24GB
    torch.cuda.empty_cache()
    torch.cuda.memory.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.8)  # Utiliser 80% de la mémoire disponible

# Configurer le gradient scaler
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'bfloat16'))

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
            X, Y = next(iter(dataset))
            with ctx:
                logits, loss = model(X, Y)
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
    wandb.init(project=wandb_project, name=wandb_run_name, config=training_config)

print("Initializing training loop...")

# Créer une queue pour la génération
generation_queue = Queue()

# Fonction de génération qui tourne dans un thread séparé
def generation_worker(model, device):
    while True:
        if not generation_queue.empty():
            encoder_input = generation_queue.get()
            with torch.no_grad():
                generated = generate_text(model, encoder_input, max_new_tokens=50, temperature=0.8)
                print(f"\nGenerated text: {generated}\n")

# Démarrer le thread de génération avant la boucle d'entraînement
generation_thread = threading.Thread(
    target=generation_worker, 
    args=(model, device),
    daemon=True
)
# generation_thread.start()

train_start_time = time.time()
total_tokens = 0
X, Y = next(iter(train_dataset)) # fetch first batch
t0 = time.time()

def extract_iter_num(filename):
    """Extrait le numéro d'itération d'un nom de fichier checkpoint."""
    try:
        return int(filename.split('ckpt_')[1].split('.pt')[0])
    except:
        return 0

def cleanup_old_checkpoints(out_dir, keep_num=3):
    """Garde uniquement les 'keep_num' checkpoints les plus récents."""
    checkpoints = glob.glob(os.path.join(out_dir, 'ckpt_*.pt'))
    if len(checkpoints) <= keep_num:
        return
        
    # Trier les checkpoints par numéro d'itération
    checkpoints.sort(key=extract_iter_num)
    
    # Supprimer les checkpoints les plus anciens
    for ckpt in checkpoints[:-keep_num]:
        try:
            os.remove(ckpt)
            print(f"Removed old checkpoint: {ckpt}")
        except Exception as e:
            print(f"Error removing checkpoint {ckpt}: {e}")

for iter_num in range(iter_num, max_iters):
    
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        console.print(f"\n[bold cyan]Step {iter_num}:[/bold cyan] val loss {losses['val']:.4f}, train loss {losses['train']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "val/loss": losses['val'],
                "train/loss": losses['train'],
                "lr": lr,
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
                    'config': training_config.__dict__,
                }
                save_path = os.path.join(out_dir, f'ckpt_{iter_num}.pt')
                print(f"saving checkpoint to {save_path}")
                torch.save(checkpoint, save_path)
                cleanup_old_checkpoints(out_dir)
    

    
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if device_type == 'cuda':
            # Transférer les tenseurs sur GPU de manière optimisée
            if X.device.type == 'cpu':
                X = X.pin_memory().to(device, non_blocking=True)
            else:
                X = X.to(device)
            if Y.device.type == 'cpu':
                Y = Y.pin_memory().to(device, non_blocking=True)
            else:
                Y = Y.to(device)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = next(iter(train_dataset))
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward() if scaler else loss.backward()
        
    if iter_num % log_interval == 0 and master_process:
        # Calculer les métriques
        dt = time.time() - t0
        tokens_per_sec = batch_size * block_size / dt
        total_tokens += batch_size * block_size
        elapsed = time.time() - train_start_time
        
        # Afficher les métriques de manière élégante
        console.print(
            f"[bold green]iter {iter_num}:[/bold green] "
            f"loss {loss.item():.4f} | "
            f"[yellow]{tokens_per_sec:.1f}[/yellow] tokens/s | "
            f"[blue]{total_tokens/1e6:.1f}M[/blue] total tokens | "
            f"lr {lr:.2e} | "
            f"time {elapsed/3600:.2f}h"
        )
    
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer) if scaler else None
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer) if scaler else optimizer.step()
    if scaler:
        scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t0 = time.time()
    
    # termination conditions
    if iter_num > max_iters:
        break

    # Dans la boucle d'entraînement, après chaque N itérations
    if iter_num % 100 == 0:  # Ajuster la fréquence selon vos besoins
        if device_type == 'cuda':
            # Forcer la synchronisation CUDA
            torch.cuda.synchronize()
            
        # Collecter manuellement le garbage
        if gc.isenabled():
            gc.collect()

    if iter_num % 100 == 0:  # Pour profiler périodiquement
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    if iter_num % 100 == 0:
        end_event.record()
        torch.cuda.synchronize()
        print(f"CUDA time: {start_event.elapsed_time(end_event)}ms")

if ddp:
    destroy_process_group()

# Nettoyage final CUDA
if device_type == 'cuda':
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, 'memory_stats'):
        torch.cuda.memory_stats(device=device)
    torch.cuda.set_stream(torch.cuda.Stream())
