"""
Training script for MoE (Mixture of Experts) encoder-decoder model.
This version is specifically optimized for training MoE architectures.
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
import sys

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp, barrier as dist_barrier
from torch.serialization import add_safe_globals

from transformers import AutoTokenizer
from model import GPTConfig
from moe import MoEEncoderDecoderGPT
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

# Ajouter après les imports
class AveragedTimingStats:
    def __init__(self, print_interval=100):
        self.timings = defaultdict(list)
        self.print_interval = print_interval
        self.current_step = 0
        self.current_context = []
        self.last_tokens = 0
        self.last_time = time.time()
        self.cycle_start_time = time.time()
        self.batch_count = 0
        
    @contextmanager
    def track(self, name):
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.timings[name].append(duration)
            
    def start_batch(self):
        self.cycle_start_time = time.time()
        self.last_tokens = 0
        
    def end_batch(self, tokens):
        batch_time = time.time() - self.cycle_start_time
        self.timings['total_batch'].append(batch_time)
        self.timings['tokens'].append(tokens)
        
    def get_throughput(self):
        total_time = sum(self.timings['total_batch'])
        total_tokens = sum(self.timings['tokens'])
        return total_tokens / total_time if total_time > 0 else 0

    def get_averaged_stats(self):
        """Calculate average timings and percentages for all tracked operations"""
        if not self.timings:
            return {}, {}

        # Calculate averages for each timing category
        avg_timings = {}
        total_time = 0.0
        for name, times in self.timings.items():
            if name == 'tokens':
                continue  # Skip token counts
            avg = sum(times) / len(times) if times else 0
            avg_timings[name] = avg
            if name != 'total_batch':  # Exclude total batch time from individual ops
                total_time += avg * len(times)

        # Calculate percentages relative to total tracked time
        percentages = {}
        for name, avg in avg_timings.items():
            if name == 'total_batch' or name == 'tokens':
                continue
            percentages[name] = (avg / total_time) * 100 if total_time > 0 else 0

        return avg_timings, percentages

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
parser = argparse.ArgumentParser(description='Train MoE model')
parser.add_argument('--size', type=str, choices=['small', 'medium'], default='small',
                   help='Model size configuration (small or medium)')
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# I/O
out_dir = 'out_moe'
eval_interval = 1000
log_interval = 1
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch' or 'resume'

# wandb logging
wandb_log = False
wandb_project = 'moe_training'
wandb_run_name = 'moe_run'

# data
dataset = 'openwebtext'
data_dir = 'data/openwebtext'
gradient_accumulation_steps = 1
dropout = 0.0
bias = False
attention_backend = "flash_attn_2" # "sdpa"

# Configure CUDA Graph behavior
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

# MoE specific configurations
if torch.cuda.is_available():
    device = f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}'
    
    # Détection automatique de la carte graphique
    gpu_name = torch.cuda.get_device_name()
    is_ampere = any(x in gpu_name.lower() for x in ['a100', 'a6000', 'a5000', 'a4000'])
    is_ada = any(x in gpu_name.lower() for x in ['4090', '4080', '4070'])
    
    if args.size == "small":
        # Increase batch size and reduce gradient accumulation steps
        batch_size = 16  
        block_size = 64
        
        num_experts = 32
        expert_k = 4
        
        gradient_accumulation_steps = 2
        
        # Encoder config
        encoder_n_layer = 4
        encoder_n_head = 4
        encoder_n_embd = 384
        encoder_ratio_kv = 4
        
        # Decoder config
        decoder_n_layer = 4
        decoder_n_head = 4
        decoder_n_embd = 384
        decoder_ratio_kv = 4
    
    elif args.size == "medium":
        # Base parameters
        block_size = 256
        num_experts = 16
        expert_k = 4
        
        
        # Ajustements spécifiques selon le GPU
        if is_ampere:
            # Optimisations A100
            batch_size = 48  # Réduit pour éviter OOM
            gradient_accumulation_steps = 2  # Augmenté pour compenser
            
            # Optimisations mémoire et calcul
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
            
            # CUDA Graph optimizations
            torch._inductor.config.triton.cudagraph_trees = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.triton.unique_kernel_names = True
            torch._inductor.config.fx_graph_cache = True
            
            # Additional A100 optimizations
            os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
            os.environ["CUDA_MODULE_LOADING"] = "LAZY"
            os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
            os.environ["NCCL_SOCKET_NTHREADS"] = "4"
            os.environ["NCCL_MIN_NCHANNELS"] = "4"
            
            # Memory management
            torch.cuda.empty_cache()
            torch.cuda.memory.set_per_process_memory_fraction(0.95)
            
            # Disable JIT cache
            torch.jit.set_fusion_strategy([('DYNAMIC', 3)])
            
            # Prefetch and overlap optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        elif is_ada:
            # Optimisations 4090
            batch_size = 18
            gradient_accumulation_steps = 2
            # La 4090 a une meilleure latence mais moins de bande passante
            torch.set_num_threads(4)  # Moins de CPU threads
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        else:
            batch_size = 16
        
        # Configuration commune
        encoder_n_layer = 8
        encoder_n_head = 8
        encoder_n_embd = 768
        encoder_ratio_kv = 8
        
        decoder_n_layer = 8
        decoder_n_head = 8
        decoder_n_embd = 768
        decoder_ratio_kv = 8
    elif args.size == "large":
        # Paramètres de base
        block_size = 256
        num_experts = 32
        expert_k = 4
        gradient_accumulation_steps = 2

        # Ajustements spécifiques selon le GPU
        if is_ampere:
            # Optimisations A100
            batch_size = 40  # Augmenté pour mieux utiliser la bande passante
            gradient_accumulation_steps = 1  # Réduit car batch_size plus grand
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Optimiser pour le throughput
            torch.set_num_threads(8)  # CPU threads pour les opérations host
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        elif is_ada:
            # Optimisations 4090
            batch_size = 32
            # La 4090 a une meilleure latence mais moins de bande passante
            torch.set_num_threads(4)  # Moins de CPU threads
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        else:
            batch_size = 24
            gradient_accumulation_steps = 3

        # Configuration commune
        encoder_n_layer = 16
        encoder_n_head = 16
        encoder_n_embd = 1024
        encoder_ratio_kv = 16
        
        decoder_n_layer = 16
        decoder_n_head = 16
        decoder_n_embd = 1024
        decoder_ratio_kv = 16
    
    router_z_loss_coef = 0.01
    router_aux_loss_coef = 0.01
    
    # Fixed sequence lengths for padding
    encoder_seq_len = block_size
    decoder_seq_len = block_size

    # CUDA Optimizations
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.cuda.set_per_process_memory_fraction(0.95)
    
    # Additional CUDA optimizations
    torch.backends.cudnn.enabled = True
    
    # Désactiver le cache JIT qui peut croître au fil du temps
    torch.jit.set_fusion_strategy([('DYNAMIC', 3)])
    
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Block size: {block_size}")
    print(f"Using {num_experts} experts with top-{expert_k} routing")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
else:
    device = 'cpu'
    batch_size = 2
    block_size = 64
    encoder_n_layer = 2
    encoder_n_head = 2
    encoder_n_embd = 64
    encoder_ratio_kv = 2
    decoder_n_layer = 2
    decoder_n_head = 2
    decoder_n_embd = 64
    decoder_ratio_kv = 2
    num_experts = 2
    expert_k = 1

# adamw optimizer
learning_rate = 5e-4
max_iters = 600000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.999
grad_clip = 0.5

# learning rate decay settings
decay_lr = True
warmup_iters = 20
lr_decay_iters = 600000 / gradient_accumulation_steps
min_lr = 6e-5

# DDP settings
backend = 'nccl'
compile = True

# Prompts for generation
PROMPT_TEMPLATES = [
    # Science Fiction
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
# Helper functions
# -----------------------------------------------------------------------------

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def reduce_metrics(metrics, world_size):
    """
    Reduces metrics across all processes.
    """
    if not isinstance(metrics, torch.Tensor):
        metrics = torch.tensor(metrics, device='cuda')
    all_reduce(metrics, op=ReduceOp.SUM)
    return metrics.item() / world_size

def calculate_perplexity(loss):
    """
    Calculates perplexity from loss
    """
    if isinstance(loss, torch.Tensor):
        loss_tensor = loss.clone().detach()
    else:
        loss_tensor = torch.tensor(loss, device='cuda')
    return torch.exp(loss_tensor).item()

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    
    # Activer le mode évaluation pour les datasets
    train_dataset.set_eval_mode(True)
    val_dataset.set_eval_mode(True)
    
    # Temporarily disable compilation for evaluation
    if hasattr(model, '_orig_mod'):
        eval_model = model._orig_mod
    else:
        eval_model = model
        
    with torch.no_grad():
        for split, dataset in [('train', train_dataset), ('val', val_dataset)]:
            losses = torch.zeros(eval_iters, device=device)
            router_losses = torch.zeros(eval_iters, device=device)
            valid_iters = 0
            
            for k in range(eval_iters):
                encoder_input, decoder_input, target = next(iter(dataset))
                with ctx:
                    logits, loss, router_loss = eval_model(encoder_input, decoder_input, target)
                    if loss is not None:
                        losses[valid_iters] = loss.item()
                        router_losses[valid_iters] = router_loss.item()
                        valid_iters += 1
            
            if valid_iters > 0:
                mean_loss = losses[:valid_iters].mean()
                mean_router_loss = router_losses[:valid_iters].mean()
                if ddp:
                    mean_loss = reduce_metrics(mean_loss, ddp_world_size)
                    mean_router_loss = reduce_metrics(mean_router_loss, ddp_world_size)
                out[split] = mean_loss
                out[f'{split}_router'] = mean_router_loss
                out[f'{split}_ppl'] = calculate_perplexity(mean_loss)
            else:
                out[split] = float('inf')
                out[f'{split}_router'] = float('inf')
                out[f'{split}_ppl'] = float('inf')
    
    # Désactiver le mode évaluation
    train_dataset.set_eval_mode(False)
    val_dataset.set_eval_mode(False)
    
    model.train()
    return out

def cleanup_old_checkpoints(out_dir, keep_num=2):
    """Keeps only the 'keep_num' most recent checkpoints."""
    checkpoints = glob.glob(os.path.join(out_dir, 'ckpt_iter_*.pt'))
    if len(checkpoints) <= keep_num:
        return
        
    def extract_iter_num(filename):
        try:
            return int(filename.split('iter_')[1].split('_loss')[0])
        except:
            return 0
            
    checkpoints.sort(key=extract_iter_num)
    
    for ckpt in checkpoints[:-keep_num]:
        try:
            os.remove(ckpt)
            print(f"Removed old checkpoint: {ckpt}")
        except Exception as e:
            print(f"Error removing checkpoint {ckpt}: {e}")

def cleanup_memory():
    """Clean up memory more aggressively"""
    if torch.cuda.is_available():
        # Synchronize all streams
        torch.cuda.synchronize()
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        
        # Force garbage collection
        gc.collect()
        
        # Clear autograd graph
        torch.cuda.synchronize()
        
        # Get memory stats for debugging
        if hasattr(torch.cuda, 'memory_stats'):
            torch.cuda.memory_stats()
        
        # Optional: clear CUDA graph cache
        if hasattr(torch.cuda, '_graph_pool_handle'):
            torch.cuda._graph_pool_handle.clear()

def ensure_model_dtype(model, dtype):
    """Ensures model is in the correct dtype"""
    for param in model.parameters():
        param.data = param.data.to(dtype=dtype)
    return model

def print_memory_stats(prefix=""):
    """Prints detailed GPU memory statistics"""
    if torch.cuda.is_available():
        print(f"\n{prefix} Memory stats:")
        print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
        print(f"Max reserved: {torch.cuda.max_memory_reserved()/1e9:.2f}GB")
        
        # Reset peak stats
        torch.cuda.reset_peak_memory_stats()

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

# -----------------------------------------------------------------------------
# Init
# -----------------------------------------------------------------------------

# Initialize distributed training
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    # NCCL Optimizations
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_NET_GDR_LEVEL"] = "2"
    os.environ["NCCL_P2P_DISABLE"] = "0"
    os.environ["NCCL_MIN_NCHANNELS"] = "4"
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    os.environ["NCCL_BUFFSIZE"] = "2097152"
    os.environ["NCCL_ALGO"] = "Ring"
    os.environ["NCCL_PROTO"] = "Simple"
    os.environ["NCCL_BLOCKING_WAIT"] = "0"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    
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

device_type = 'cuda' if 'cuda' in device else 'cpu'
dtype = 'bfloat16' if attention_backend == "flash_attn_2" else 'float16'
print(f"Using dtype: {dtype}")
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Initialize tokenizer
access_token = os.getenv('HF_TOKEN')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_fast=True, access_token=access_token)
tokenizer.pad_token = tokenizer.eos_token

# Initialize model configs
vocab_size = len(tokenizer)

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

model_args = {
    'encoder_config': encoder_config,
    'decoder_config': decoder_config,
    'num_experts': num_experts,
    'expert_k': expert_k
}

# Model initialization
iter_num = 1
best_val_loss = float('inf')

# Ajouter la définition du scaler ici
scaler = torch.amp.GradScaler(enabled=(dtype == 'bfloat16' or dtype == 'float16'))

if init_from == 'resume':
    print(f"Attempting to resume training from {out_dir}")
    checkpoints = glob.glob(os.path.join(out_dir, 'ckpt_iter_*.pt'))
    if not checkpoints:
        print("No checkpoints found, initializing from scratch instead")
        init_from = 'scratch'
    else:
        def extract_iter_num(filename):
            try:
                return int(filename.split('iter_')[1].split('_loss')[0])
            except:
                return 0
        
        checkpoints.sort(key=extract_iter_num)
        ckpt_path = checkpoints[-1]
        print(f"Loading checkpoint: {ckpt_path}")
        
        # Charger sur CPU d'abord
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Vérifier que les configurations correspondent
        assert checkpoint['model_args']['num_experts'] == num_experts, "Number of experts mismatch"
        assert checkpoint['model_args']['expert_k'] == expert_k, "Expert k mismatch"
        
        # Créer un nouveau modèle
        model = MoEEncoderDecoderGPT(
            encoder_config,
            decoder_config,
            num_experts=num_experts,
            k=expert_k
        )
        
        # Nettoyer le state dict
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        cleaned_state_dict = {
            k[len(unwanted_prefix):] if k.startswith(unwanted_prefix) else k: v
            for k, v in state_dict.items()
        }
        
        # Charger le state dict
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        
        # Déplacer le modèle sur le device
        model = model.to(device)
        
        # Créer un nouvel optimiseur
        optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
        
        # Charger l'état de l'optimiseur si présent
        if 'optimizer' in checkpoint:
            optimizer_state = checkpoint['optimizer']
            # S'assurer que tous les tenseurs sont sur le bon device
            for state in optimizer_state['state'].values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device=device)
            optimizer.load_state_dict(optimizer_state)
        
        # Réinitialiser complètement le scaler
        scaler = torch.amp.GradScaler(enabled=(dtype == 'bfloat16' or dtype == 'float16'))
        
        iter_num = checkpoint['iter_num'] + 1
        best_val_loss = checkpoint['best_val_loss']
        
        # Nettoyage
        del checkpoint, state_dict, cleaned_state_dict
        if 'optimizer_state' in locals():
            del optimizer_state
        torch.cuda.empty_cache()

if init_from == 'scratch':
    print("Initializing a new MoE encoder-decoder model from scratch")
    model = MoEEncoderDecoderGPT(
        encoder_config,
        decoder_config,
        num_experts=num_experts,
        k=expert_k
    )
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# Compile model if requested
if compile:
    print("Compiling model...")
    model.set_gradient_checkpointing(True)  # Activé pour économiser la mémoire
    try:
        model = torch.compile(
            model,
            mode="reduce-overhead",
            # options={
            #     "max_autotune": True,
            #     "epilogue_fusion": True,
            #     "triton.cudagraphs": True,  # Activé
            #     "trace.graph_diagram": False,
            #     # Ajouter des options pour mieux gérer les formes dynamiques
            #     # "dynamic_memory": True,
            #     # "max_parallel_block_sizes": 6,
            #     "max_autotune_gemm": True,
            # }
        )
    except Exception as e:
        print(f"Compilation failed: {e}")
        print(traceback.format_exc())
        compile = False

# Remplacez timing_stats = TimingStats() par:
timing_stats = AveragedTimingStats(print_interval=100)  # Afficher les stats tous les 10 iterations
# Après la création du modèle et avant de le déplacer sur le device
model.set_timing_stats(timing_stats)

# Si vous utilisez DDP, modifiez aussi:
if ddp:
    model = DDP(
        model,
        device_ids=[ddp_local_rank],
        output_device=ddp_local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True
    )
    # Assurez-vous que timing_stats est accessible dans le modèle wrappé
    model.module.set_timing_stats(timing_stats)

model.to(device)

# CUDA optimizations
if device_type == 'cuda':
    print("CUDA optimizations")
    default_stream = torch.cuda.current_stream()
    copy_stream = torch.cuda.Stream()
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    device_index = 0 if isinstance(device, str) else device
    if isinstance(device, str) and ':' in device:
        device_index = int(device.split(':')[1])
    torch.cuda.set_device(device_index)
    torch.cuda.empty_cache()
    
    pin_memory = True
    
    # gc.enable()
    
    torch.cuda.set_per_process_memory_fraction(0.95)

    if is_ampere:
        # Optimisations spécifiques A100
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cudnn.benchmark = True
        
        # Optimiser pour le throughput
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        torch.cuda.set_per_process_memory_fraction(0.98)
        
        # Configuration des streams
        torch.cuda.Stream(priority=-1)
        
        # Optimisation de la mémoire
        torch.cuda.memory.set_per_process_memory_fraction(0.98)
        torch.cuda.memory.set_per_process_memory_fraction(0.98, 0)
        
        # Désactiver le garbage collector pendant l'entraînement
        gc.disable()
    elif is_ada:
        # Optimisations spécifiques 4090
        # Privilégier la latence
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"
        torch.cuda.set_per_process_memory_fraction(0.85)

# Configure gradient scaler
scaler = torch.amp.GradScaler(enabled=(dtype == 'bfloat16' or dtype == 'float16'))

# Initialize datasets
train_dataset, val_dataset = get_datasets(block_size, batch_size, device)
train_iterator = iter(train_dataset)

# Initialize timing
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0
train_start_time = time.time()
tokens_per_iter = batch_size * block_size * gradient_accumulation_steps
total_tokens = 0

print(f"Starting training with {num_experts} experts and top-{expert_k} routing")
print_memory_stats("Initial")

# Simplify the forward/backward step:
def forward_backward_step():
    try:
        with torch.no_grad():
            inputs = next(train_iterator)
        
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            _, loss, router_loss = model(*inputs)
            scaler.scale(loss + 0.01*router_loss).backward()
            
        return loss.item(), router_loss.item()
    except Exception as e:
        print(f"Step failed: {e}")
        return 0.0, 0.0

# training loop
while True:
    # print("Starting training loop, iter_num:", iter_num)
    # determine and set the learning rate for this iteration
    with timing_stats.track("lr_update"):
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process and iter_num > 0:
        with timing_stats.track("validation"):
            print("Validation")
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, "
                  f"train router loss {losses['train_router']:.4f}, "
                  f"train ppl {losses['train_ppl']:.2f}, "
                  f"val loss {losses['val']:.4f}, "
                  f"val router loss {losses['val_router']:.4f}, "
                  f"val ppl {losses['val_ppl']:.2f}")

            if losses['val'] < best_val_loss or always_save_checkpoint:
                with timing_stats.track("checkpoint"):
                    best_val_loss = losses['val']
                    if iter_num > 0:
                        checkpoint = {
                            'model': {k: v.cpu() for k, v in model.state_dict().items()},
                            'optimizer': {
                                'state': {k: {sk: sv.cpu() if isinstance(sv, torch.Tensor) else sv 
                                         for sk, sv in v.items()}
                                         for k, v in optimizer.state_dict()['state'].items()},
                                'param_groups': optimizer.state_dict()['param_groups']
                            },
                            'model_args': model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                        }
                        checkpoint_path = os.path.join(
                            out_dir,
                            f'ckpt_iter_{iter_num}_loss_{losses["val"]:.4f}.pt'
                        )
                        torch.save(checkpoint, checkpoint_path)
                        cleanup_old_checkpoints(out_dir)
                        del checkpoint
                        cleanup_memory()

    # Generate text every 200 iterations
    if iter_num % 500 == 0 and master_process:
        print("\nGénération de texte :")
        prompt = random.choice(PROMPT_TEMPLATES)
        
        # Tokenize with proper handling of special tokens
        input_tokens = tokenizer.encode(
            prompt,
            add_special_tokens=True,
            truncation=False,  # Ne pas tronquer le prompt
            padding=False,     # Ne pas ajouter de padding
            return_tensors='pt'
        ).to(device)
        
        # Vérifier que nous avons bien tout le prompt
        decoded_prompt = tokenizer.decode(input_tokens[0], skip_special_tokens=True)
        print(f"Prompt original: {prompt}")
        print(f"Prompt décodé: {decoded_prompt}")
        
        try:
            with torch.no_grad(), torch.amp.autocast(enabled=True, device_type=device_type):
                # Generate text
                output_ids = model.generate(
                    input_tokens,
                    max_new_tokens=block_size - 10,  # Slightly reduced for better focus
                    temperature=0.7,     # Reduced for more coherence
                    top_k=40            # Reduced for more focused sampling
                )
                
                # Decode only the generated part
                generated_text = tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                print(f"Generated text: {generated_text}")
        except Exception as e:
            print(f"Erreur lors de la génération: {str(e)}")
            # print the stack trace
            print(traceback.format_exc())
            continue

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with gradient accumulation
    with timing_stats.track("optimization"):
        optimizer.zero_grad(set_to_none=False)
        total_loss = 0
        total_router_loss = 0
        batch_tokens = 0  # Track total tokens for this batch
        
        timing_stats.start_batch()
        
        try:
            for micro_step in range(gradient_accumulation_steps):
                # Add NCCL synchronization check
                if ddp:
                    torch.distributed.barrier(timeout=torch.timedelta(seconds=30))
                
                model.require_backward_grad_sync = (
                    micro_step == gradient_accumulation_steps - 1
                )
                
                # Get losses and perform backward pass
                loss_val, router_loss_val = forward_backward_step()
                if loss_val is None:
                    continue
                
                # Accumulate scalar values
                total_loss += loss_val
                total_router_loss += router_loss_val
                
                # Accumulate tokens (per microstep)
                micro_batch_tokens = batch_size * block_size  # Tokens per microstep
                batch_tokens += micro_batch_tokens  # Sum across microsteps
                
                # End of batch processing - call once per batch
                timing_stats.end_batch(batch_tokens)  # Pass total tokens for full batch

        except Exception as e:
            print(f"Training failed at iter {iter_num}: {e}")
            if ddp:
                torch.distributed.destroy_process_group()
            sys.exit(1)


    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    if iter_num % log_interval == 0:
        lossf = total_loss
        router_lossf = total_router_loss
        
        # Get detailed timing breakdown
        avg_timings, percentages = timing_stats.get_averaged_stats()
        
        # Calculate precise tokens per second
        tps = timing_stats.get_throughput()
        total_tps = sum(timing_stats.timings['tokens']) / sum(timing_stats.timings['total_batch'])
        
        if local_iter_num >= 5:
            mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        
        print(f"iter {iter_num}: loss {lossf:.4f}, router_loss {router_lossf:.4f}, "
              f"time {dt*1000:.2f}ms, lr {lr:.2e}, t/s {tps:.2f}, avg t/s {total_tps:.2f}")
        
        # Print timing breakdown
     

    iter_num += 1
    local_iter_num += 1
    
    if iter_num % 50 == 0:  # Monitoring plus fréquent
        print("\nTiming breakdown:")
        for name in ['data_loading', 'forward', 'backward', 'optimizer_step']:
            if name in avg_timings:
                print(f"{name}: {avg_timings[name]*1000:.1f}ms ({percentages[name]:.1f}%)")
        
    #     print_memory_stats(f"Iter {iter_num}")
    #     cleanup_memory()

    # termination conditions
    if iter_num > max_iters:
        break

    if iter_num % 10 == 0:
        print(f"Training alive check - iter {iter_num}")
        if device_type == 'cuda':
            print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
            torch.cuda.synchronize()  # Force CUDA sync
            

if ddp:
    destroy_process_group()

# Final cleanup
if device_type == 'cuda':
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# Add memory optimizations after model initialization
if device_type == 'cuda':
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

if master_process:
    print("Training finished!")

# At the top of the file:
torch._inductor.config.triton.cudagraph_trees = False  # Disable tree reduction
torch._inductor.config.triton.store_cubin = False  # Disable cubin caching

