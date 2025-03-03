"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).
"""

import os
import time
import pickle
from contextlib import nullcontext
import threading
from queue import Queue
import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group, dist_barrier
from torch.serialization import add_safe_globals

from transformers import AutoTokenizer
import pickle

from model import GPTConfig, GPT, EncoderDecoderGPT
from data.openwebtext.data_loader import StreamingDataset
from run_train import get_datasets
from train_utils import (
    get_gpu_count, setup_distributed, reduce_metrics, calculate_perplexity,
    get_lr, cleanup_old_checkpoints, cleanup_memory, ensure_model_dtype,
    print_memory_stats, setup_cuda_optimizations, save_checkpoint,
    load_checkpoint, find_latest_checkpoint, get_context_manager,
    generate_text, evaluate_model, estimate_loss
)

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
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
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
attention_backend = "sdpa" # Force specific attention backend (flash_attn_2, xformers, sdpa, or None for auto)

# Configurations pour l'encoder et le decoder
if torch.cuda.is_available():
    device = f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}'  # Use LOCAL_RANK for DDP
    # batch_size = 16 # Réduire la taille du batch
    # block_size = 512
    
    batch_size = 16
    block_size = 64
    
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Block size: {block_size}")
    
    # Encoder config (plus petit)
    encoder_n_layer = 8
    encoder_n_head = 8  # 1024/16 = 64 par tête
    encoder_n_embd = 768  # Doit être divisible par encoder_n_head
    encoder_ratio_kv = 8  # 16/4 = 4 têtes KV
    
    # Decoder config (plus grand)
    decoder_n_layer = 12
    decoder_n_head = 12  # 1024/32 = 32 par tête
    decoder_n_embd = 768  # Doit être divisible par decoder_n_head
    decoder_ratio_kv = 8  # 32/4 = 8 têtes KV
    
    
    # Optimisations mémoire
    # gradient_accumulation_steps = max(1, 64 // (batch_size * torch.cuda.device_count()))  # Adjust for multi-GPU
    gradient_accumulation_steps = 8  # Augmenter l'accumulation
    
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    
    # Activer la gestion de mémoire optimisée
    setup_cuda_optimizations()
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
warmup_iters = 1000 / gradient_accumulation_steps # how many steps to warm up for
lr_decay_iters = 600000 / gradient_accumulation_steps # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    ddp_rank, ddp_local_rank, ddp_world_size, device = setup_distributed(backend=backend)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
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
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
dtype = 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = get_context_manager(device_type, dtype)

print(f"Device type: {device_type}")
print(f"PT dtype: {ptdtype}")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_fast=True, access_token=access_token)
tokenizer.pad_token = tokenizer.eos_token

decode = lambda tokens: tokenizer.decode(tokens, skip_special_tokens=True)

# init these up here, can override if init_from='resume'
iter_num = 1
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
    dropout=dropout,
    attention_backend=attention_backend
)

decoder_config = GPTConfig(
    n_layer=decoder_n_layer,
    n_head=decoder_n_head,
    n_embd=decoder_n_embd,
    block_size=block_size,
    ratio_kv=decoder_ratio_kv,
    bias=bias,
    vocab_size=vocab_size,
    dropout=dropout,
    attention_backend=attention_backend
)

# Store model arguments for checkpointing
model_args = {
    'encoder_config': encoder_config,
    'decoder_config': decoder_config,
    'vocab_size': vocab_size,
    'block_size': block_size
}

# Ajouter après les imports, avant le chargement du checkpoint
add_safe_globals([GPTConfig])

if init_from == 'resume':
    print(f"Attempting to resume training from {out_dir}")
    ckpt_path = find_latest_checkpoint(out_dir)
    if not ckpt_path:
        print("No checkpoints found, initializing from scratch instead")
        init_from = 'scratch'
    else:
        print(f"Loading checkpoint: {ckpt_path}")
        model = EncoderDecoderGPT(encoder_config, decoder_config)
        model, optimizer, iter_num, best_val_loss, _, _ = load_checkpoint(
            ckpt_path, model, map_location='cpu'
        )
        if optimizer is None:
            optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
        
        # Ensure model parameters are in the correct dtype
        model = ensure_model_dtype(model, ptdtype)
        cleanup_memory()

if init_from == 'scratch':
    print("Initializing a new encoder-decoder model from scratch")
    model = EncoderDecoderGPT(encoder_config, decoder_config)
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# Juste avant le model.to(device):
if compile:
    print("Compiling the model with reduced optimization...")
    compile_options = {
        "mode": "max-autotune",
        # "fullgraph": False,
        # "dynamic": True,
        # "backend": "inductor",
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

# Configurer le gradient scaler
scaler = torch.amp.GradScaler(enabled=(dtype == 'bfloat16' or dtype == 'float16'))

# wrap model into DDP container
if ddp:
    model = DDP(
        model, 
        device_ids=[ddp_local_rank],
        output_device=ddp_local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,  # Garder True
        gradient_as_bucket_view=True
    )

    # Ajouter un hook pour déboguer les paramètres inutilisés si nécessaire
    if os.environ.get('TORCH_DISTRIBUTED_DEBUG', '').upper() in ('INFO', 'DETAIL'):
        def print_unused_parameters(model):
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"Parameter without gradient: {name}")
        
        def hook(state):
            print_unused_parameters(model.module)
        
        model.register_comm_hook(None, hook)

# Initialize datasets
train_dataset, val_dataset = get_datasets(block_size, batch_size, device)
print(f"vocab_size: {len(train_dataset.tokenizer)}")

# Get vocab size from the dataset
vocab_size = len(train_dataset.tokenizer)
print(f"vocab_size: {vocab_size}")

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

print("Initializing training loop...")

# Créer une queue pour la génération
generation_queue = Queue()

# Fonction de génération qui tourne dans un thread séparé
def generation_worker():
    while True:
        if not generation_queue.empty():
            encoder_input = generation_queue.get()
            with torch.no_grad():
                generated = generate_text(model, encoder_input, max_new_tokens=50, temperature=0.8, tokenizer=tokenizer)
                print(f"\nGenerated text: {generated}\n")

# Démarrer le thread de génération avant la boucle d'entraînement
generation_thread = threading.Thread(
    target=generation_worker, 
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

print_memory_stats("Initial")

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process and iter_num > 0:
        print("Validation")
        # Use utility function for loss estimation
        losses = estimate_loss(model, train_dataset, val_dataset, eval_iters, device, ctx, ddp, ddp_world_size)
        
        print(f"step {iter_num}: train loss {losses['train']:.4f}, train ppl {losses['train_ppl']:.2f}, "
              f"val loss {losses['val']:.4f}, val ppl {losses['val_ppl']:.2f}")
        print(f"Total tokens processed: {train_dataset.get_total_tokens():,}")
        print_memory_stats("After validation")
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "train/perplexity": losses['train_ppl'],
                "val/loss": losses['val'],
                "val/perplexity": losses['val_ppl'],
                "lr": lr,
                "mfu": running_mfu*100,
                "total_tokens": train_dataset.get_total_tokens()
            })
            
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint_path = save_checkpoint(
                    raw_model, 
                    optimizer, 
                    model_args, 
                    iter_num, 
                    best_val_loss, 
                    config,
                    out_dir,
                    losses['val']
                )
                cleanup_memory()
                cleanup_old_checkpoints(out_dir)
                
        # Generate examples for evaluation
        if iter_num % generate_interval == 0 and master_process:
            print("\nGenerating examples:")
            example_results = evaluate_model(
                raw_model, 
                val_dataset, 
                ctx, 
                num_examples=2, 
                max_tokens=50, 
                temperature=0.8,
                tokenizer=tokenizer
            )
            
            for i, result in enumerate(example_results):
                print(f"Example {i+1}:")
                print(f"Input: {result['input']}")
                print(f"Generated: {result['generated']}")
                print("-" * 50)
                
            if wandb_log:
                wandb_examples = []
                for result in example_results:
                    wandb_examples.append([result['input'], result['generated']])
                wandb.log({
                    "examples": wandb.Table(
                        columns=["Input", "Generated"],
                        data=wandb_examples
                    )
                })
                
    if iter_num == 0 and eval_only:
        break

    # Continue avec le code normal d'entraînement
    optimizer.zero_grad(set_to_none=True)
    skip_optimizer_step = False
    
    # Déclarer les variables pour la prochaine itération
    encoder_input_next, decoder_input_next, target_next = None, None, None
    loss = None

    try:
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # Only synchronize gradients on the last micro step
                model.require_backward_grad_sync = (
                    micro_step == gradient_accumulation_steps - 1
                )
            
            try:
                encoder_input_next, decoder_input_next, target_next = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataset)
                encoder_input_next, decoder_input_next, target_next = next(train_iterator)
            
            # Forward pass with error handling
            try:
                with ctx:
                    logits, current_loss = model(encoder_input, decoder_input, target)
                    if current_loss is not None:
                        # Scale loss for gradient accumulation
                        current_loss = current_loss / gradient_accumulation_steps
                        # Scale for mixed precision
                        scaled_loss = scaler.scale(current_loss)
                        # Backward pass
                        scaled_loss.backward()
                        loss = current_loss
            except RuntimeError as e:
                import traceback
                print("\n=== Error Stack Trace ===")
                print(f"Process rank: {ddp_rank if ddp else 'N/A'}")
                print(f"Error: {str(e)}")
                print("\nFull stack trace:")
                print(traceback.format_exc())
                print("=" * 50)
                skip_optimizer_step = True
                break

            
            # Move data preparation for next iteration here
            encoder_input, decoder_input, target = (
                encoder_input_next, 
                decoder_input_next,
                target_next
            )
            
        # Optimizer step if no errors occurred
        if not skip_optimizer_step and loss is not None:
            if scaler.is_enabled() and scaler._scale is not None:
                # Unscale gradients and clip
                if grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                # Step with scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                # Normal optimizer step
                if grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            
            # Zero gradients after optimizer step
            optimizer.zero_grad(set_to_none=True)
            
    except Exception as e:
        print(f"Training iteration failed: {e}")
        if ddp:
            dist_barrier()  # Synchronize processes before continuing
        continue

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    # Mettre à jour total_tokens avant la réduction
    step_tokens = (batch_size * block_size) * gradient_accumulation_steps
    total_tokens += step_tokens

    if iter_num % log_interval == 0:
        # Calculer les métriques locales (par GPU)
        lossf = loss.item() * gradient_accumulation_steps if loss is not None else 0.0
        elapsed = time.time() - train_start_time
        local_dt = dt
        local_tokens = total_tokens
        local_tps = step_tokens / dt

        # Métriques agrégées (tous les GPUs)
        if ddp:
            # Créer des tensors sur le GPU pour la réduction
            global_loss = torch.tensor(lossf, device=device)
            global_dt = torch.tensor(dt, device=device)
            global_tokens = torch.tensor(total_tokens, device=device)
            
            # Réduire les métriques à travers tous les processus
            all_reduce(global_loss, op=ReduceOp.SUM)
            all_reduce(global_dt, op=ReduceOp.SUM)
            all_reduce(global_tokens, op=ReduceOp.SUM)
            
            # Calculer les moyennes
            global_loss = global_loss.item() / ddp_world_size
            global_dt = global_dt.item() / ddp_world_size
            global_tokens = global_tokens.item()  # Ne pas diviser les tokens, on veut le total
            global_tps = global_tokens / elapsed
        else:
            global_loss = lossf
            global_dt = dt
            global_tokens = local_tokens
            global_tps = local_tps

        # Afficher les logs uniquement sur le processus maître
        if master_process:
            # Log global (tous les GPUs combinés)
            print(
                f"iter_num: {iter_num}, "
                f"loss: {global_loss:.4f}, "
                f"total_tps: {global_tps:.1f} t/s, "
                f"total_tokens: {global_tokens/1e6:.2f}M, "
                f"time/step: {global_dt*1000:.2f}ms, "
                f"time: {elapsed/60:.2f}min, "
                f"Total GPU: {ddp_world_size}"
            )
            

            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "loss": global_loss,
                    "tokens_per_sec/gpu": local_tps,
                    "tokens_per_sec/total": global_tps,
                    "tokens/gpu": local_tokens,
                    "tokens/total": global_tokens,
                    "learning_rate": lr,
                    "step_time_ms": local_dt * 1000
                })

    iter_num += 1
    local_iter_num += 1
    encoder_input, decoder_input, target = encoder_input_next, decoder_input_next, target_next
    
    # termination conditions
    if iter_num > max_iters:
        break

    # Dans la boucle d'entraînement, après chaque N itérations
    if iter_num % 100 == 0:  # Ajuster la fréquence selon vos besoins
        if device_type == 'cuda':
            # Forcer la synchronisation CUDA
            torch.cuda.synchronize()
            
        # Collecter manuellement le garbage
        cleanup_memory()

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
cleanup_memory()

# Après l'initialisation des datasets
if master_process:
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    # Vérifier un échantillon
    sample_encoder, sample_decoder, sample_target = next(iter(train_dataset))
    print(f"Sample batch shapes - encoder: {sample_encoder.shape}, decoder: {sample_decoder.shape}, target: {sample_target.shape}")
