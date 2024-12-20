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


from model import GPTConfig, GPT, EncoderDecoderGPT
from data.openwebtext.data_loader import StreamingDataset
from run_train import get_datasets

from rich.console import Console
console = Console()

access_token=os.getenv('HF_TOKEN')

# Ajouter l'import au début du fichier
from pytorch_optimizer import AdEMAMix

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
if torch.cuda.is_available():
    device = f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}'  # Use LOCAL_RANK for DDP
    batch_size = 32 # Réduire la taille du batch
    block_size = 512
    
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Block size: {block_size}")
    
    # Encoder config (plus petit)
    encoder_n_layer = 4
    encoder_n_head = 8
    encoder_n_embd = 768
    encoder_ratio_kv = 8
    
    # Decoder config (plus grand)
    decoder_n_layer = 12
    decoder_n_head = 12
    decoder_n_embd = 768 
    decoder_ratio_kv = 8
    
    # Optimisations mémoire
    # gradient_accumulation_steps = max(1, 64 // (batch_size * torch.cuda.device_count()))  # Adjust for multi-GPU
    gradient_accumulation_steps = 16  # Augmenter l'accumulation
    
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    dtype = 'bfloat16'
    
    # Activer la gestion de mémoire optimisée
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Activer la mémoire partagée
    if hasattr(torch.cuda, 'memory_stats'):
        torch.cuda.memory_stats()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))  # Set device based on LOCAL_RANK
else:
    device = 'cpu'
    batch_size = 2
    block_size = 64
    
    # Encoder config (minimal)
    encoder_n_layer = 1
    encoder_n_head = 1
    encoder_n_embd = 2
    encoder_ratio_kv = 1
    
    # Decoder config (minimal)
    decoder_n_layer = 1
    decoder_n_head = 1
    decoder_n_embd = 2
    decoder_ratio_kv = 1


# adamw optimizer
learning_rate = 1e-3 # ajusté pour AdEMAMix
max_iters = 600000 # total number of training iterations
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.999
beta3 = 0.9999  # nouveau paramètre pour AdEMAMix
alpha = 8.0  # nouveau paramètre pour AdEMAMix
beta3_warmup = alpha_warmup = 256_000  # périodes de warmup pour AdEMAMix
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 / gradient_accumulation_steps # how many steps to warm up for
lr_decay_iters = 600000 / gradient_accumulation_steps # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # Optimisations NCCL
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_NET_GDR_LEVEL"] = "2"
    os.environ["NCCL_P2P_DISABLE"] = "0"
    os.environ["NCCL_MIN_NCHANNELS"] = "4"
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    os.environ["NCCL_BUFFSIZE"] = "2097152"
    
    # Réduire la fréquence de synchronisation
    os.environ["NCCL_ALGO"] = "Ring"
    os.environ["NCCL_PROTO"] = "Simple"
    
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
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

# Choisir manuellement dtype = "float16" si BF16 n'est pas supporté
dtype = "float16"
ptdtype = torch.float16

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Instancier ou charger le modèle
model = EncoderDecoderGPT(encoder_config, decoder_config)

# Juste avant le model.to(device):
if compile:
    print("Compiling the model with reduced optimization...")
    compile_options = {
        "mode": "reduce-overhead",
        "fullgraph": False,
        "dynamic": True,
        "backend": "inductor",
        "options": {
            "allow_rnn": True,
            "allow_gradient_checkpointing": True
        }
    }
    
    # Disable gradient checkpointing before compilation
    model.set_gradient_checkpointing(False)
    
    try:
        model = torch.compile(model, **compile_options)
        print(f"Model compiled with mode: {compile_options['mode']}")
    except Exception as e:
        print(f"Warning: Model compilation failed, continuing without compilation: {e}")
        compile = False
        # Re-enable gradient checkpointing
        model.set_gradient_checkpointing(True)

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
    
    # Optimisations mémoire supplémentaires
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    
    # Configurer la pagination de la mémoire
    torch.cuda.set_per_process_memory_fraction(0.95)
    
    # Désactiver le gradient synchrone pour DDP
    os.environ["NCCL_BLOCKING_WAIT"] = "0"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

# Configurer le gradient scaler
scaler = torch.amp.GradScaler(enabled=(dtype == 'bfloat16' or dtype == 'float16'))

# optimizer
#  optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2, beta3), device_type)
optimizer = model.configure_optimizers_adamw(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# wrap model into DDP container
if ddp:
    model = DDP(
        model, 
        device_ids=[ddp_local_rank],
        find_unused_parameters=True,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        static_graph=True,
        bucket_cap_mb=25  # Augmenter la taille des buckets
    )

# Initialize datasets
train_dataset, val_dataset = get_datasets(block_size, batch_size, device)
print(f"vocab_size: {len(train_dataset.tokenizer)}")

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
        
    # Trier les checkpoints par numéro d'it��ration
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

# Ajouter ces imports au début du fichier
from torch.profiler import profile, record_function, ProfilerActivity
import time

# Ajouter cette fonction après la fonction estimate_loss()
def profile_training_step(model, encoder_input, decoder_input, target, optimizer, scaler, ctx):
    """Profile une étape d'entraînement complète"""
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True
    ) as prof:
        # Forward pass
        with record_function("forward"):
            with ctx:
                try:
                    logits, loss = model(encoder_input, decoder_input, target)
                except Exception as e:
                    print(f"Forward pass failed: {e}")
                    return None
        
        # Backward pass
        with record_function("backward"):
            if loss is not None:
                try:
                    scaled_loss = scaler.scale(loss)
                    scaled_loss.backward()
                except Exception as e:
                    print(f"Backward pass failed: {e}")
                    return None
        
        # Optimizer step
        with record_function("optimizer"):
            try:
                if scaler.is_enabled() and scaler._scale is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad(set_to_none=True)
            except Exception as e:
                print(f"Optimizer step failed: {e}")
                return None
    
    return prof

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

    # Profiling périodique (toutes les 100 itérations)
    if iter_num % 100 == 0 and master_process:
        # print("\n=== Starting profiling for iteration", iter_num, "===")
        # prof = profile_training_step(
        #     model, encoder_input, decoder_input, target,
        #     optimizer, scaler, ctx
        # )
        prof = None
        
        if prof is not None:
            print("\nProfiling results:")
            print(prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=10))
            
            print("\nMemory stats:")
            print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
            print(f"Reserved memory: {torch.cuda.memory_reserved() / 1024**2:.1f}MB")
            print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.1f}MB")
            
            # Temps par étape
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            # Une étape d'entraînement normale
            with ctx:
                try:
                    logits, loss = model(encoder_input, decoder_input, target)
                    if loss is not None:
                        scaled_loss = scaler.scale(loss)
                        scaled_loss.backward()
                except Exception as e:
                    print(f"Error during timing measurement: {e}")
            end_event.record()
            
            torch.cuda.synchronize()
            step_time = start_event.elapsed_time(end_event)
            print(f"\nSingle step time: {step_time:.2f}ms")
            print("=" * 40 + "\n")
        else:
            print("Profiling failed")
        
    # Continue avec le code normal d'entraînement
    optimizer.zero_grad(set_to_none=True)
    skip_optimizer_step = False
    
    # Déclarer les variables pour la prochaine itération
    encoder_input_next, decoder_input_next, target_next = None, None, None
    loss = None

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # Ne synchroniser qu'à la dernière étape d'accumulation
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        
        with torch.amp.autocast(enabled=True, dtype=ptdtype, device_type=device_type):
            with torch.cuda.stream(copy_stream):
                try:
                    encoder_input_next, decoder_input_next, target_next = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_dataset)
                    encoder_input_next, decoder_input_next, target_next = next(train_iterator)
            
            # Attendre que les données soient prêtes
            torch.cuda.current_stream().wait_stream(copy_stream)
            
            # Forward pass
            logits, current_loss = model(encoder_input, decoder_input, target)
            if current_loss is not None:
                # Scale loss
                current_loss = current_loss / gradient_accumulation_steps
                
                # Backward pass
                scaled_loss = scaler.scale(current_loss)
                scaled_loss.backward()
                
                # Mettre à jour loss pour le logging
                loss = current_loss
        
        # # Préparer la prochaine itération : charger le batch suivant
        # try:
        #     encoder_input_next, decoder_input_next, target_next = next(train_iterator)
        # except StopIteration:
        #     train_iterator = iter(train_dataset)
        #     encoder_input_next, decoder_input_next, target_next = next(train_iterator)

    if not skip_optimizer_step:
        if scaler.is_enabled() and scaler._scale is not None:
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        
        lossf = loss.item() * gradient_accumulation_steps
        dt = time.time() - t0
        elapsed = time.time() - train_start_time
        total_tokens += (batch_size * block_size) * gradient_accumulation_steps
        tokens_per_sec = total_tokens / elapsed
        
        # Afficher les métriques de manière élégante
        # console.print(
        #     f"[bold green]{iter_num}:[/bold green] "
        #     f"loss {loss.item():.4f} | "
        #     f"[yellow]{tokens_per_sec:.1f}[/yellow] t/s | "
        #     f"[blue]{total_tokens/1e6:.1f}M[/blue] total t | "
        #     f"lr {lr:.2e} | "
        #     f"time {dt:.2f}s | "
        #     f"total {elapsed/60:.2f}min"
        # )
        
        # Simple print sans coloration
        print(f"iter_num: {iter_num}, loss: {lossf:.4f}, tps: {tokens_per_sec:.1f} t/s, tt: {total_tokens/1e3:.1f}K, lr: {lr:.2e}, time: {dt*1000:.4f}ms, total: {elapsed/60:.2f}min")
        
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

# Après l'initialisation des datasets
if master_process:
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    # Vérifier un échantillon
    sample_encoder, sample_decoder, sample_target = next(iter(train_dataset))
    print(f"Sample batch shapes - encoder: {sample_encoder.shape}, decoder: {sample_decoder.shape}, target: {sample_target.shape}")
