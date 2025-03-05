"""
Script d'entraînement générique pour les modèles LLM.
Ce script peut être utilisé pour entraîner différents types de modèles (DeepSeek, LLaDA, etc.)
"""

import os
import time
import math
import gc
import glob
import threading
import argparse
import traceback
import random


import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group, barrier as dist_barrier
from torch.amp import GradScaler

# Import des fonctions d'optimisation GPU
from gpu_optimization import (
    setup_cuda_optimizations, cleanup_memory, print_memory_stats, 
    print_gpu_stats, preallocate_cuda_memory
)

# Import des fonctions utilitaires
from train.train_utils import (
    get_gpu_count, setup_distributed, reduce_metrics, calculate_perplexity,
    get_lr, cleanup_old_checkpoints, ensure_model_dtype,
    save_checkpoint, load_checkpoint, find_latest_checkpoint, 
    get_context_manager, AveragedTimingStats, generate_text
)

class Trainer:
    """
    Classe générique pour l'entraînement des modèles LLM.
    """
    def __init__(self, args):
        """
        Initialise le trainer avec les arguments fournis.
        
        Args:
            args: Arguments de configuration pour l'entraînement
        """
        self.args = args
        self.setup_environment()
        self.setup_model()
        self.setup_datasets()
        self.setup_training()
        
    def setup_environment(self):
        """Configure l'environnement d'exécution (DDP, device, etc.)"""
        # Distributed setup
        self.ddp = int(os.environ.get('RANK', -1)) != -1
        if self.ddp:
            self.ddp_rank, self.ddp_local_rank, self.ddp_world_size, self.device = setup_distributed(backend=self.args.backend)
            self.master_process = self.ddp_rank == 0
            self.seed_offset = self.ddp_rank
        else:
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1
            self.device = self.args.device if hasattr(self.args, 'device') else 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # Create output directory
        if self.master_process:
            os.makedirs(self.args.output_dir, exist_ok=True)
        
        # Set random seed
        torch.manual_seed(1337 + self.seed_offset)
        
        # Setup device and dtype
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        
        # Determine dtype based on model type and available hardware
        if hasattr(self.args, 'dtype'):
            self.dtype = self.args.dtype
        else:
            self.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
            
        print(f"Using dtype: {self.dtype}")
        self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = get_context_manager(self.device_type, self.dtype)
        
        # Apply CUDA optimizations if available
        if torch.cuda.is_available():
            setup_cuda_optimizations()
            if hasattr(self.args, 'preallocate_memory') and self.args.preallocate_memory:
                preallocate_cuda_memory()
        
        # Calculate tokens per iteration for logging
        self.tokens_per_iter = self.args.batch_size * self.args.block_size * self.args.gradient_accumulation_steps * self.ddp_world_size
        print(f"Tokens per iteration: {self.tokens_per_iter:,}")
        
    def setup_model(self):
        """Initialise le modèle en fonction du type spécifié"""
        # Determine model type and initialize
        model_type = self.args.model_type.lower()
        
        if self.args.init_from == 'resume':
            print(f"Attempting to resume training from {self.args.output_dir}")
            ckpt_path = find_latest_checkpoint(self.args.output_dir)
            if not ckpt_path:
                print("No checkpoints found, initializing from scratch instead")
                self.args.init_from = 'scratch'
            else:
                print(f"Loading checkpoint: {ckpt_path}")
                self.load_model_from_checkpoint(ckpt_path)
                return
        
        # Initialize from scratch
        print(f"Initializing a new {model_type} model from scratch")
        
        if model_type == 'deepseek':
            from deepseek import DeepSeekMini, DeepSeekMiniConfig
            # Create config based on model size
            config = self.create_deepseek_config()
            self.model = DeepSeekMini(config)
            
        elif model_type == 'llada':
            from models.llada.model import LLaDAModel, LLaDAConfig
            # Create config based on model size
            config = self.create_llada_config()
            self.model = LLaDAModel(config)
            
        else:
            from models.models.model import GPT, GPTConfig
            # Create config based on model size
            config = self.create_gpt_config()
            self.model = GPT(config)
        
        # Store config for checkpointing
        self.config = config
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self.model.configure_optimizers(
            weight_decay=self.args.weight_decay,
            learning_rate=self.args.learning_rate,
            betas=(self.args.beta1, self.args.beta2),
            device_type=self.device_type
        )
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler(enabled=(self.dtype == 'bfloat16' or self.dtype == 'float16'))
        
        # Initialize training state
        self.iter_num = 1
        self.best_val_loss = float('inf')
        
        # Print model size
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")
        
    def create_deepseek_config(self):
        """Crée une configuration pour le modèle DeepSeek Mini"""
        from deepseek import DeepSeekMiniConfig
        
        # Determine model size parameters
        if self.args.size == 'small':
            config = DeepSeekMiniConfig(
                vocab_size=self.args.vocab_size,
                hidden_size=1024,
                num_hidden_layers=8,
                num_attention_heads=8,
                head_dim=64,
                intermediate_size=2816,
                num_experts=8,
                num_experts_per_token=2,
                max_position_embeddings=self.args.block_size,
                kv_compression_dim=64,
                query_compression_dim=192,
                rope_head_dim=32,
                dropout=self.args.dropout,
                attention_dropout=self.args.dropout,
                hidden_dropout=self.args.dropout,
                bias=self.args.bias
            )
        elif self.args.size == 'medium':
            config = DeepSeekMiniConfig(
                vocab_size=self.args.vocab_size,
                hidden_size=2048,
                num_hidden_layers=24,
                num_attention_heads=16,
                head_dim=128,
                intermediate_size=4096,
                num_experts=32,
                num_experts_per_token=4,
                max_position_embeddings=self.args.block_size,
                kv_compression_dim=128,
                query_compression_dim=384,
                rope_head_dim=32,
                dropout=self.args.dropout,
                attention_dropout=self.args.dropout,
                hidden_dropout=self.args.dropout,
                bias=self.args.bias
            )
        else:  # large
            config = DeepSeekMiniConfig(
                vocab_size=self.args.vocab_size,
                hidden_size=3072,
                num_hidden_layers=32,
                num_attention_heads=24,
                head_dim=128,
                intermediate_size=8192,
                num_experts=64,
                num_experts_per_token=4,
                max_position_embeddings=self.args.block_size,
                kv_compression_dim=256,
                query_compression_dim=768,
                rope_head_dim=32,
                dropout=self.args.dropout,
                attention_dropout=self.args.dropout,
                hidden_dropout=self.args.dropout,
                bias=self.args.bias
            )
        
        return config
    
    def create_llada_config(self):
        """Crée une configuration pour le modèle LLaDA"""
        from models.llada.model import LLaDAConfig
        
        # Determine model size parameters
        if self.args.size == 'small':
            config = LLaDAConfig(
                block_size=self.args.block_size,
                vocab_size=self.args.vocab_size,
                n_layer=8,
                n_head=8,
                n_embd=768,
                dropout=self.args.dropout,
                bias=self.args.bias,
                ratio_kv=8,
                use_checkpoint=False
            )
        elif self.args.size == 'medium':
            config = LLaDAConfig(
                block_size=self.args.block_size,
                vocab_size=self.args.vocab_size,
                n_layer=12,
                n_head=12,
                n_embd=1024,
                dropout=self.args.dropout,
                bias=self.args.bias,
                ratio_kv=8,
                use_checkpoint=False
            )
        else:  # large
            config = LLaDAConfig(
                block_size=self.args.block_size,
                vocab_size=self.args.vocab_size,
                n_layer=24,
                n_head=16,
                n_embd=1536,
                dropout=self.args.dropout,
                bias=self.args.bias,
                ratio_kv=8,
                use_checkpoint=False
            )
        
        return config
    
    def create_gpt_config(self):
        """Crée une configuration pour le modèle GPT standard"""
        from models.models.model import GPTConfig
        
        # Determine model size parameters
        if self.args.size == 'small':
            config = GPTConfig(
                n_layer=8,
                n_head=8,
                n_embd=768,
                block_size=self.args.block_size,
                bias=self.args.bias,
                vocab_size=self.args.vocab_size,
                dropout=self.args.dropout,
                attention_backend=self.args.attention_backend if hasattr(self.args, 'attention_backend') else None
            )
        elif self.args.size == 'medium':
            config = GPTConfig(
                n_layer=12,
                n_head=12,
                n_embd=1024,
                block_size=self.args.block_size,
                bias=self.args.bias,
                vocab_size=self.args.vocab_size,
                dropout=self.args.dropout,
                attention_backend=self.args.attention_backend if hasattr(self.args, 'attention_backend') else None
            )
        else:  # large
            config = GPTConfig(
                n_layer=24,
                n_head=16,
                n_embd=1536,
                block_size=self.args.block_size,
                bias=self.args.bias,
                vocab_size=self.args.vocab_size,
                dropout=self.args.dropout,
                attention_backend=self.args.attention_backend if hasattr(self.args, 'attention_backend') else None
            )
        
        return config
    
    def load_model_from_checkpoint(self, ckpt_path):
        """Charge un modèle à partir d'un checkpoint"""
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Determine model type
        model_type = self.args.model_type.lower()
        
        if model_type == 'deepseek':
            from deepseek import DeepSeekMini, DeepSeekMiniConfig
            # Create new model with saved config
            saved_config = DeepSeekMiniConfig(**checkpoint['model_args'])
            self.model = DeepSeekMini(saved_config)
            self.config = saved_config
            
        elif model_type == 'llada':
            from models.llada.model import LLaDAModel, LLaDAConfig
            # Create new model with saved config
            if 'model_args' in checkpoint and isinstance(checkpoint['model_args'], dict):
                if 'llada_config' in checkpoint['model_args']:
                    llada_config_dict = checkpoint['model_args']['llada_config']
                    saved_config = LLaDAConfig(**llada_config_dict)
                else:
                    # Fallback for older checkpoints
                    saved_config = LLaDAConfig(**checkpoint['model_args'])
            else:
                # Create default config
                saved_config = self.create_llada_config()
                
            self.model = LLaDAModel(saved_config)
            self.config = saved_config
            
        else:
            from models.models.model import GPT, GPTConfig
            # Create new model with saved config
            if 'model_args' in checkpoint and isinstance(checkpoint['model_args'], dict):
                saved_config = GPTConfig(**checkpoint['model_args'])
            else:
                # Create default config
                saved_config = self.create_gpt_config()
                
            self.model = GPT(saved_config)
            self.config = saved_config
        
        # Load model and optimizer states
        self.model, self.optimizer, self.iter_num, self.best_val_loss, _, _ = load_checkpoint(
            ckpt_path, self.model, map_location='cpu'
        )
        
        if self.optimizer is None:
            self.optimizer = self.model.configure_optimizers(
                weight_decay=self.args.weight_decay,
                learning_rate=self.args.learning_rate,
                betas=(self.args.beta1, self.args.beta2),
                device_type=self.device_type
            )
        
        # Move model to device and ensure correct dtype
        self.model = self.model.to(self.device)
        self.model = ensure_model_dtype(self.model, self.ptdtype)
        
        # Reset scaler
        self.scaler = GradScaler(enabled=(self.dtype == 'bfloat16' or self.dtype == 'float16'))
        
        cleanup_memory()
    
    def setup_datasets(self):
        """Configure les datasets d'entraînement et de validation"""
        # Import the get_datasets function from run_train
        from run_train import get_datasets
        
        # Get datasets
        if hasattr(self.args, 'tokenizer'):
            self.train_dataset, self.val_dataset = get_datasets(
                self.args.block_size, 
                self.args.batch_size, 
                self.device, 
                tokenizer=self.args.tokenizer
            )
        else:
            self.train_dataset, self.val_dataset = get_datasets(
                self.args.block_size, 
                self.args.batch_size, 
                self.device
            )
        
        # Create iterator
        self.train_iterator = iter(self.train_dataset)
        
    def setup_training(self):
        """Configure les paramètres d'entraînement"""
        # Initialize timing stats
        self.timing_stats = AveragedTimingStats(print_interval=100)
        if hasattr(self.model, 'set_timing_stats'):
            self.model.set_timing_stats(self.timing_stats)
        
        # Initialize DDP if needed
        if self.ddp:
            self.model = DDP(
                self.model,
                device_ids=[self.ddp_local_rank],
                output_device=self.ddp_local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True
            )
            if hasattr(self.model.module, 'set_timing_stats'):
                self.model.module.set_timing_stats(self.timing_stats)
        
        # Compile model if requested
        if hasattr(self.args, 'compile') and self.args.compile:
            print("Compiling model...")
            if hasattr(self.model, 'set_gradient_checkpointing'):
                self.model.set_gradient_checkpointing(True)
            try:
                self.model = torch.compile(self.model)
            except Exception as e:
                print(f"Compilation failed: {e}")
                print(traceback.format_exc())
                self.args.compile = False
        
        # Initialize timing variables
        self.t0 = time.time()
        self.local_iter_num = 0
        self.running_mfu = -1.0
        self.train_start_time = time.time()
        self.total_tokens = 0
        self.tokens_window = []  # Pour calculer une moyenne glissante des tokens/s
        self.window_size = 10   # Taille de la fenêtre pour la moyenne glissante
        
        # Print training info
        model_type = self.args.model_type.lower()
        print(f"Starting training {model_type} model")
        if model_type == 'deepseek':
            print(f"Model size: {self.config.hidden_size}d, {self.config.num_hidden_layers}l, {self.config.num_attention_heads}h")
            print(f"Using {self.config.num_experts} experts with top-{self.config.num_experts_per_token} routing")
        elif model_type == 'llada':
            print(f"Model size: {self.config.n_embd}d, {self.config.n_layer}l, {self.config.n_head}h")
        else:
            print(f"Model size: {self.config.n_embd}d, {self.config.n_layer}l, {self.config.n_head}h")
            
        print(f"Batch size: {self.args.batch_size}, Block size: {self.args.block_size}")
        print(f"Gradient accumulation steps: {self.args.gradient_accumulation_steps}")
        print_memory_stats("Initial")
    
    def train(self):
        """Exécute la boucle d'entraînement principale"""
        model_type = self.args.model_type.lower()
        
        # Training loop
        while True:
            # Determine and set the learning rate for this iteration
            with self.timing_stats.track("lr_update"):
                lr = get_lr(
                    self.iter_num, 
                    self.args.warmup_iters, 
                    self.args.lr_decay_iters, 
                    self.args.learning_rate, 
                    self.args.min_lr
                ) if self.args.decay_lr else self.args.learning_rate
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            
            # Generate text periodically
            if self.iter_num % 100 == 0 and self.master_process:
                self.generate_sample_text()
            
            # Evaluate model periodically
            if self.iter_num % self.args.eval_interval == 0 and self.master_process and self.iter_num > 0:
                self.evaluate_model()
            
            # Exit if eval_only is set
            if self.iter_num == 0 and self.args.eval_only:
                break
            
            # Forward backward update, with gradient accumulation
            with self.timing_stats.track("optimization"):
                self.optimizer.zero_grad(set_to_none=True)
                total_loss = 0
                total_router_loss = 0
                skip_optimizer_step = False
                
                try:
                    for micro_step in range(self.args.gradient_accumulation_steps):
                        if self.ddp:
                            self.model.require_backward_grad_sync = (
                                micro_step == self.args.gradient_accumulation_steps - 1
                            )
                        
                        # Track data loading time
                        with self.timing_stats.track("data_loading"):
                            try:
                                batch = next(self.train_iterator)
                            except StopIteration:
                                self.train_iterator = iter(self.train_dataset)
                                batch = next(self.train_iterator)
                            
                            # Handle different batch formats
                            if isinstance(batch, dict):
                                input_ids = batch['input_ids'].to(self.device)
                                targets = batch.get('labels', input_ids).to(self.device)
                            elif isinstance(batch, tuple) and len(batch) >= 2:
                                input_ids = batch[0].to(self.device)
                                targets = batch[-1].to(self.device)
                            else:
                                input_ids = batch.to(self.device)
                                targets = batch.to(self.device)
                        
                        # Forward pass
                        with self.timing_stats.track("forward"), torch.amp.autocast(enabled=True, device_type=self.device_type):
                            # Handle different model types
                            if model_type == 'deepseek':
                                # DeepSeek model returns a dict
                                outputs = self.model(
                                    input_ids=input_ids,
                                    attention_mask=None,  # Will be created automatically if needed
                                )
                                
                                # Unpack outputs
                                hidden_states = outputs["last_hidden_state"]
                                balance_loss = outputs.get("balance_loss", 0)
                                
                                # Calculate loss
                                raw_model = self.model.module if self.ddp else self.model
                                logits = hidden_states @ raw_model.embed_tokens.weight.t()
                                shift_logits = logits[..., :-1, :].contiguous()
                                shift_labels = targets[..., 1:].contiguous()
                                
                                loss = F.cross_entropy(
                                    shift_logits.view(-1, shift_logits.size(-1)),
                                    shift_labels.view(-1),
                                    ignore_index=-1
                                )
                                
                                # Track tokens
                                batch_tokens = input_ids.ne(0).sum().item()  # Assuming 0 is pad token
                                
                            elif model_type == 'llada':
                                # LLaDA model
                                logits, loss, router_loss = self.model(input_ids, targets=targets)
                                balance_loss = router_loss if router_loss is not None else 0
                                
                                # Track tokens
                                batch_tokens = input_ids.numel()
                                
                            else:
                                # Standard GPT model
                                logits, loss = self.model(input_ids, targets=targets)
                                balance_loss = 0
                                
                                # Track tokens
                                batch_tokens = input_ids.numel()
                            
                            # Update token counts
                            self.total_tokens += batch_tokens
                            self.tokens_window.append((time.time(), batch_tokens))
                            if len(self.tokens_window) > self.window_size:
                                self.tokens_window.pop(0)
                            
                            # Clean up to save memory
                            if 'logits' in locals():
                                del logits
                            if 'hidden_states' in locals():
                                del hidden_states
                        
                        # Backward pass
                        with self.timing_stats.track("backward"):
                            if loss is not None:
                                # Scale loss for gradient accumulation
                                loss = loss / self.args.gradient_accumulation_steps
                                
                                # Add router loss if available
                                if balance_loss != 0:
                                    balance_loss = balance_loss / self.args.gradient_accumulation_steps
                                    router_loss_coef = getattr(self.args, 'router_z_loss_coef', 0.001)
                                    combined_loss = loss + router_loss_coef * balance_loss
                                else:
                                    combined_loss = loss
                                
                                # Backward pass with scaler
                                scaled_loss = self.scaler.scale(combined_loss)
                                scaled_loss.backward()
                                
                                # Track losses
                                total_loss += loss.item()
                                if balance_loss != 0:
                                    total_router_loss += balance_loss.item()
                                
                                # Clean up
                                del loss
                                if balance_loss != 0:
                                    del balance_loss
                                del combined_loss
                                del scaled_loss
                
                except Exception as e:
                    print(f"Training iteration failed: {e}")
                    print(traceback.format_exc())
                    cleanup_memory()
                    if self.ddp:
                        dist_barrier()
                    continue
                
                # Optimizer step
                with self.timing_stats.track("optimizer_step"):
                    if self.args.grad_clip != 0.0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters() if not self.ddp else self.model.module.parameters(),
                            self.args.grad_clip
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            
            # Timing and logging
            t1 = time.time()
            dt = t1 - self.t0
            self.t0 = t1
            
            if self.iter_num % self.args.log_interval == 0:
                self.log_training_stats(total_loss, total_router_loss, dt, lr)
                
                # Save checkpoint periodically
                if self.iter_num % 1000 == 0 and self.master_process:
                    self.save_training_checkpoint()
            
            self.iter_num += 1
            self.local_iter_num += 1
            
            # Periodic memory cleanup
            if self.iter_num % 100 == 0:
                cleanup_memory()
            
            # Termination conditions
            if self.iter_num > self.args.max_iters:
                break
        
        # Cleanup
        if self.ddp:
            destroy_process_group()
        
        cleanup_memory()
        
        if self.master_process:
            print("Training finished!")
    
    def generate_sample_text(self):
        """Génère un exemple de texte avec le modèle actuel"""
        model_type = self.args.model_type.lower()
        
        print("\nText Generation:")
        
        # Get a prompt
        if hasattr(self.args, 'prompt_templates') and self.args.prompt_templates:
            prompt = random.choice(self.args.prompt_templates)
        else:
            prompt = "Once upon a time"
        
        # Tokenize the prompt
        if hasattr(self.args, 'tokenizer'):
            tokenizer = self.args.tokenizer
            input_tokens = tokenizer.encode(
                prompt,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_tensors='pt'
            ).to(self.device)
        else:
            # Simple fallback if no tokenizer
            input_tokens = torch.tensor([[1, 2, 3]]).to(self.device)  # Dummy tokens
        
        try:
            with torch.no_grad(), torch.amp.autocast(enabled=True, device_type=self.device_type):
                # Generate text
                raw_model = self.model.module if self.ddp else self.model
                
                if model_type == 'deepseek':
                    output_ids = raw_model.generate(
                        input_tokens,
                        max_new_tokens=min(100, self.args.block_size - 10),
                        temperature=0.7,
                        top_k=40
                    )
                else:
                    # Use the generic generate_text function
                    output_text = generate_text(
                        raw_model,
                        input_tokens,
                        max_new_tokens=min(100, self.args.block_size - 10),
                        temperature=0.7,
                        tokenizer=tokenizer if hasattr(self.args, 'tokenizer') else None
                    )
                    print(f"Generated text: {prompt} {output_text}\n")
                    return
                
                # Decode the generated text
                if hasattr(self.args, 'tokenizer'):
                    generated_text = tokenizer.decode(
                        output_ids[0],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    print(f"Generated text: {prompt} {generated_text}\n")
                else:
                    print(f"Generated tokens: {output_ids[0].tolist()}\n")
                    
        except Exception as e:
            print(f"Generation error: {str(e)}")
            print(traceback.format_exc())
    
    def evaluate_model(self):
        """Évalue le modèle sur les ensembles d'entraînement et de validation"""
        from train_utils import estimate_loss
        
        print("Validation")
        # Use utility function for loss estimation
        losses = estimate_loss(
            self.model, 
            self.train_dataset, 
            self.val_dataset, 
            self.args.eval_iters, 
            self.device, 
            self.ctx, 
            self.ddp, 
            self.ddp_world_size
        )
        
        print(f"step {self.iter_num}: train loss {losses['train']:.4f}, train ppl {losses['train_ppl']:.2f}, "
              f"val loss {losses['val']:.4f}, val ppl {losses['val_ppl']:.2f}")
        
        # Log to wandb if enabled
        if hasattr(self.args, 'wandb_log') and self.args.wandb_log:
            import wandb
            wandb.log({
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "train/perplexity": losses['train_ppl'],
                "val/perplexity": losses['val_ppl'],
                # Include router losses if available
                "train/router_loss": losses.get('train_router_loss', 0.0),
                "val/router_loss": losses.get('val_router_loss', 0.0),
                "lr": self.optimizer.param_groups[0]['lr'],
                "iter": self.iter_num,
                "total_tokens": self.total_tokens
            })
        
        # Save checkpoint if best validation loss
        if losses['val'] < self.best_val_loss or self.args.always_save_checkpoint:
            self.best_val_loss = losses['val']
            if self.iter_num > 0:
                self.save_training_checkpoint(val_loss=losses['val'])
    
    def log_training_stats(self, total_loss, total_router_loss, dt, lr):
        """Affiche les statistiques d'entraînement"""
        lossf = total_loss
        router_lossf = total_router_loss
        
        # Calculate tokens/s on the sliding window
        if len(self.tokens_window) > 1:
            window_time = self.tokens_window[-1][0] - self.tokens_window[0][0]
            window_tokens = sum(tokens for _, tokens in self.tokens_window)
            current_tokens_per_sec = window_tokens / window_time if window_time > 0 else 0
        else:
            current_tokens_per_sec = 0
        
        total_time = time.time() - self.train_start_time
        avg_tokens_per_sec = self.total_tokens / total_time if total_time > 0 else 0
        
        # Calculate MFU if model supports it
        if hasattr(self.model, 'estimate_mfu') and self.local_iter_num >= 5:
            raw_model = self.model.module if self.ddp else self.model
            mfu = raw_model.estimate_mfu(self.args.batch_size * self.args.gradient_accumulation_steps, dt)
            self.running_mfu = mfu if self.running_mfu == -1.0 else 0.9*self.running_mfu + 0.1*mfu
            mfu_str = f", mfu {self.running_mfu*100:.2f}%"
        else:
            mfu_str = ""
        
        # Print stats
        if router_lossf > 0:
            print(f"iter {self.iter_num}: loss {lossf:.4f}, router_loss {router_lossf:.4f}, "
                  f"time {dt*1000:.2f}ms, lr {lr:.2e}, "
                  f"tt {self.total_tokens:,}, t/s {current_tokens_per_sec:.2f}, "
                  f"avgt/s {avg_tokens_per_sec:.2f}{mfu_str}")
        else:
            print(f"iter {self.iter_num}: loss {lossf:.4f}, "
                  f"time {dt*1000:.2f}ms, lr {lr:.2e}, "
                  f"tt {self.total_tokens:,}, t/s {current_tokens_per_sec:.2f}, "
                  f"avgt/s {avg_tokens_per_sec:.2f}{mfu_str}")
        
        # Update timing stats
        self.timing_stats.step()
        
        # Print timing stats if needed
        if self.timing_stats.should_print():
            self.timing_stats.print_stats()
        
        # Log to wandb if enabled
        if hasattr(self.args, 'wandb_log') and self.args.wandb_log:
            import wandb
            wandb.log({
                "iter": self.iter_num,
                "loss": lossf,
                "router_loss": router_lossf,
                "tokens_per_sec": current_tokens_per_sec,
                "avg_tokens_per_sec": avg_tokens_per_sec,
                "learning_rate": lr,
                "step_time_ms": dt * 1000
            })
    
    def save_training_checkpoint(self, val_loss=None):
        """Sauvegarde un checkpoint d'entraînement"""
        raw_model = self.model.module if self.ddp else self.model
        
        # Get model args based on model type
        model_type = self.args.model_type.lower()
        if model_type == 'deepseek':
            model_args = self.config.__dict__
        else:
            model_args = self.config.__dict__
        
        # Save checkpoint
        checkpoint_path = save_checkpoint(
            raw_model,
            self.optimizer,
            model_args,
            self.iter_num,
            self.best_val_loss,
            vars(self.args),
            self.args.output_dir,
            val_loss
        )
        
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Cleanup
        cleanup_memory()
        cleanup_old_checkpoints(self.args.output_dir, keep_num=self.args.keep_checkpoints if hasattr(self.args, 'keep_checkpoints') else 3)


def main():
    """Point d'entrée principal pour l'entraînement"""
    # Ce code est appelé lorsque train.py est exécuté directement
    # Il est préférable d'utiliser run_train.py comme point d'entrée
    print("Please use run_train.py as the entry point for training.")
    print("Example: python run_train.py --model_type deepseek --size small")


if __name__ == '__main__':
    main()
