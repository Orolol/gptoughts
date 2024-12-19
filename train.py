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
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp

from model import GPTConfig, GPT, EncoderDecoderGPT
from data.openwebtext.data_loader import StreamingDataset

from rich.console import Console
console = Console()

access_token=os.getenv('HF_TOKEN')

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 1000
log_interval = 1
generate_interval = 100
eval_iters = 100
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
def setup_device_config():
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            ddp_world_size = n_gpus
            device = 'cuda'
        else:
            device = 'cuda:0'
            ddp_world_size = 1
            
        batch_size = 32 // ddp_world_size  # Ajuster le batch size par GPU
        block_size = 512
        
        # Encoder config
        encoder_n_layer = 4
        encoder_n_head = 8
        encoder_n_embd = 768
        encoder_ratio_kv = 8
        
        # Decoder config
        decoder_n_layer = 12
        decoder_n_head = 12
        decoder_n_embd = 768
        decoder_ratio_kv = 8
        
        # Optimisations mémoire
        gradient_accumulation_steps = 1
        dtype = 'bfloat16'
        
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    else:
        device = 'cpu'
        ddp_world_size = 1
        batch_size = 2
        block_size = 64
        
        # Configs minimales
        encoder_n_layer = encoder_n_head = 1
        encoder_n_embd = encoder_ratio_kv = 2
        decoder_n_layer = decoder_n_head = 1
        decoder_n_embd = decoder_ratio_kv = 2
        
    return (device, ddp_world_size, batch_size, block_size, 
            encoder_n_layer, encoder_n_head, encoder_n_embd, encoder_ratio_kv,
            decoder_n_layer, decoder_n_head, decoder_n_embd, decoder_ratio_kv)

# Fonction pour initialiser le processus DDP
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

# Fonction principale d'entraînement
def train(rank, world_size, model_args):
    setup_ddp(rank, world_size)
    
    # Déplacer le modèle sur le GPU approprié
    torch.cuda.set_device(rank)
    model = EncoderDecoderGPT(**model_args).to(rank)
    model = DDP(model, device_ids=[rank])
    
    # ... reste de la logique d'entraînement ...

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
        # Gnérer token par token
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
            try:
                return int(filename.split('iter_')[1].split('_loss')[0])
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

# training loop
train_iterator = iter(train_dataset)
encoder_input, decoder_input, target = next(train_iterator)
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

print("Starting training...")


train_start_time = time.time()
total_tokens = 0
t0 = time.time()

def cleanup_old_checkpoints(out_dir, keep_num=3):
    """Garde uniquement les 'keep_num' checkpoints les plus récents."""
    checkpoints = glob.glob(os.path.join(out_dir, 'ckpt_iter_*.pt'))
    if len(checkpoints) <= keep_num:
        return
        
    # Trier les checkpoints par numéro d'itération
    def extract_iter_num(filename):
        try:
            return int(filename.split('iter_')[1].split('_loss')[0])
        except:
            return 0
            
    checkpoints.sort(key=extract_iter_num)
    
    # Supprimer les checkpoints les plus anciens
    for ckpt in checkpoints[:-keep_num]:
        try:
            os.remove(ckpt)
            print(f"Removed old checkpoint: {ckpt}")
        except Exception as e:
            print(f"Error removing checkpoint {ckpt}: {e}")

def setup_memory_management(rank):
    if torch.cuda.is_available():
        # Configurer la mémoire pour ce GPU spécifique
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()
        
        # Réserver environ 80% de la mémoire disponible sur ce GPU
        total_memory = torch.cuda.get_device_properties(rank).total_memory
        torch.cuda.set_per_process_memory_fraction(0.8, rank)
        
        # Activer les optimisations CUDA
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

def parse_args():
    parser = argparse.ArgumentParser(description='Training script with configurable batch size')
    parser.add_argument('--batch_size', type=int, default=None, 
                       help='Batch size for training. If not specified, uses default configuration.')
    parser.add_argument('--multi_gpu', action='store_true',
                       help='Enable automatic multi-GPU training')
    return parser.parse_args()

args = parse_args()

if torch.cuda.is_available():
    device = 'cuda:0'
    # Utiliser la batch_size des arguments si spécifiée, sinon utiliser la valeur par défaut
    batch_size = args.batch_size if args.batch_size is not None else 32
    
    if args.multi_gpu:
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print(f"Using {n_gpus} GPUs")
            batch_size = batch_size // n_gpus  # Ajuster le batch size par GPU
            print(f"Adjusted batch size per GPU: {batch_size}")
    
    block_size = 512
    
    # Encoder config (plus petit)
    encoder_n_layer = 4
    encoder_n_head = 8
    encoder_n_embd = 768
    encoder_ratio_kv = 8
    
    # Decoder config
    decoder_n_layer = 12
    decoder_n_head = 12
    decoder_n_embd = 768
    decoder_ratio_kv = 8
    
    # Optimisations mémoire
    gradient_accumulation_steps = 1
    dtype = 'bfloat16'
    
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
else:
    device = 'cpu'
    # Même logique pour CPU
    batch_size = args.batch_size if args.batch_size is not None else 2
    block_size = 64
    # ... (reste de la configuration CPU)

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process and iter_num > 0:
        print("Validation")
        losses = estimate_loss()
        # print the first token of the input
        
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
                checkpoint_path = os.path.join(
                    out_dir, 
                    f'ckpt_iter_{iter_num}_loss_{losses["val"]:.4f}.pt'
                )
                print(f"saving checkpoint to {checkpoint_path}")
                torch.save(checkpoint, checkpoint_path)
                # Nettoyer les anciens checkpoints
                cleanup_old_checkpoints(out_dir, keep_num=3)
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
        
        # Print the first token of the input
        # print(f"First token of the input: {tokenizer.decode(encoder_input[0].tolist())}")
        
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        

        lossf = loss.item() * gradient_accumulation_steps
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
            f"time {elapsed/60:.2f}min"
        )
        
    if iter_num % generate_interval == 0 and master_process:
        if generation_queue.empty():  # Pour éviter d'accumuler des requêtes
            generation_queue.put(encoder_input.detach().clone())
    iter_num += 1
    local_iter_num += 1
    encoder_input, decoder_input, target = encoder_input_next, decoder_input_next, target_next
    
    # if iter_num % 100 == 0:
    #     generated = generate_text(model, encoder_input, max_new_tokens=50, temperature=0.8)
    #     print(f"\nGenerated text Q: {generated}\n")

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

if __name__ == '__main__':
    # Obtenir la configuration
    (device, ddp_world_size, batch_size, block_size,
     encoder_n_layer, encoder_n_head, encoder_n_embd, encoder_ratio_kv,
     decoder_n_layer, decoder_n_head, decoder_n_embd, decoder_ratio_kv) = setup_device_config()

    if ddp_world_size > 1:
        # Lancer l'entraînement multi-GPU
        mp.spawn(
            train,
            args=(ddp_world_size, model_args),
            nprocs=ddp_world_size,
            join=True
        )
    else:
        # Entraînement single-GPU ou CPU
        train(0, 1, model_args)
