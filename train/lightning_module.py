import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset
from torch.amp import GradScaler
import os
import time
import traceback
import random

# Import necessary components from your project
from models.deepseek.deepseek_adapter_mtp import DeepSeekMiniMTP, DeepSeekMiniConfigMTP
from models.llada.model import LLaDAModel, LLaDAConfig
from models.models.model import GPT, GPTConfig
from models.models.mla_model import MLAModel, MLAModelConfig, create_mla_model
from train.train_utils import (
    get_lr, calculate_perplexity, ensure_model_dtype,
    AveragedTimingStats, generate_text, estimate_loss
)
from optimization.memory_optim import cleanup_memory, print_memory_stats
from optimization.cuda_optim import setup_cuda_optimizations, print_gpu_stats

# Try importing advanced optimizations, fallback if not available
try:
    from optimization.cuda_optim import setup_cuda_optimizations, print_gpu_stats, optimize_attention_operations
    from optimization.memory_optim import cleanup_memory, print_memory_stats
    ENHANCED_OPTIMIZATIONS = True
except ImportError:
    from optimization.cuda_optim import setup_cuda_optimizations, print_gpu_stats
    from optimization.memory_optim import cleanup_memory, print_memory_stats
    ENHANCED_OPTIMIZATIONS = False


class LLMLightningModule(pl.LightningModule):
    """PyTorch Lightning Module for training LLMs."""

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args) # Saves args to self.hparams
        self.args = args # Keep args accessible directly too
        self.model = self._build_model()
        self.timing_stats = AveragedTimingStats(print_interval=1000)
        self.train_start_time = time.time()
        self.total_tokens = 0
        self.tokens_window = []
        self.window_size = 10
        self.running_mfu = -1.0

        # Apply CUDA optimizations if available and requested
        if torch.cuda.is_available():
            setup_cuda_optimizations()
            if ENHANCED_OPTIMIZATIONS and hasattr(self.args, 'optimize_attention') and self.args.optimize_attention:
                optimize_attention_operations()
            if hasattr(self.args, 'preallocate_memory') and self.args.preallocate_memory:
                preallocate_cuda_memory()
            if self.global_rank == 0:
                 print_gpu_stats()

        # Print model size
        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"Initialized {self.args.model_type} model ({self.args.size}) with {param_count:.2f}M parameters.")
        print_memory_stats("After Model Init")

    def _build_model(self):
        """Initializes the model based on configuration."""
        model_type = self.args.model_type.lower()
        print(f"Building model: {model_type} size: {self.args.size}")

        # Determine config based on model type and size
        if model_type == 'deepseek':
            config = self._create_deepseek_config()
            model = DeepSeekMiniMTP(config)
        elif model_type == 'llada':
            config = self._create_llada_config()
            model = LLaDAModel(config)
        elif model_type == 'mla':
            config = self._create_mla_config()
            model = self._create_mla_model(config)
        else: # gpt
            config = self._create_gpt_config()
            model = GPT(config)

        self.config = config # Store config for potential use later
        return model

    # --- Config Creation Methods (Copied from Trainer) ---
    def _create_deepseek_config(self):
        from models.deepseek import DeepSeekMiniConfig
        if self.args.size == 'small':
            config = DeepSeekMiniConfigMTP(
                vocab_size=self.args.vocab_size, hidden_size=1024, num_hidden_layers=8,
                num_attention_heads=8, head_dim=128, intermediate_size=2816,
                num_experts=4, num_experts_per_token=1,
                max_position_embeddings=max(16, self.args.block_size),
                kv_compression_dim=64, query_compression_dim=192, rope_head_dim=32,
                dropout=self.args.dropout, attention_dropout=self.args.dropout,
                hidden_dropout=self.args.dropout, bias=self.args.bias
            )
        elif self.args.size == 'medium':
             config = DeepSeekMiniConfigMTP(
                vocab_size=self.args.vocab_size, hidden_size=2048, num_hidden_layers=24,
                num_attention_heads=16, head_dim=128, intermediate_size=4096,
                num_experts=32, num_experts_per_token=4,
                max_position_embeddings=self.args.block_size,
                kv_compression_dim=128, query_compression_dim=384, rope_head_dim=32,
                dropout=self.args.dropout, attention_dropout=self.args.dropout,
                hidden_dropout=self.args.dropout, bias=self.args.bias
            )
        else:  # large
            config = DeepSeekMiniConfigMTP(
                vocab_size=self.args.vocab_size, hidden_size=3072, num_hidden_layers=32,
                num_attention_heads=24, head_dim=128, intermediate_size=8192,
                num_experts=64, num_experts_per_token=4,
                max_position_embeddings=self.args.block_size,
                kv_compression_dim=256, query_compression_dim=768, rope_head_dim=32,
                dropout=self.args.dropout, attention_dropout=self.args.dropout,
                hidden_dropout=self.args.dropout, bias=self.args.bias
            )
        return config

    def _create_llada_config(self):
        from models.llada.model import LLaDAConfig
        if self.args.size == 'small':
            config = LLaDAConfig(
                block_size=self.args.block_size, vocab_size=self.args.vocab_size,
                n_layer=8, n_head=8, n_embd=768, dropout=self.args.dropout,
                bias=self.args.bias, ratio_kv=8, use_checkpoint=False
            )
        elif self.args.size == 'medium':
            config = LLaDAConfig(
                block_size=self.args.block_size, vocab_size=self.args.vocab_size,
                n_layer=16, n_head=16, n_embd=1024, dropout=self.args.dropout,
                bias=self.args.bias, ratio_kv=8, use_checkpoint=False
            )
        else: # large
            config = LLaDAConfig(
                block_size=self.args.block_size, vocab_size=self.args.vocab_size,
                n_layer=24, n_head=16, n_embd=1536, dropout=self.args.dropout,
                bias=self.args.bias, ratio_kv=8, use_checkpoint=False
            )
        return config

    def _create_mla_config(self):
        """Create configuration for MLA-Model."""
        # Define key parameters based on size
        if self.args.size == 'small':
            n_layer = 12
            n_embd = 768
            n_head = 12
        elif self.args.size == 'medium':
            n_layer = 24
            n_embd = 1024
            n_head = 16
        elif self.args.size == 'large':
            n_layer = 32
            n_embd = 2048
            n_head = 16
        else:  # xl
            n_layer = 40
            n_embd = 2560
            n_head = 20
        
        # Create config object
        config = MLAModelConfig(
            # Architecture
            n_layer=n_layer,
            n_embd=n_embd,
            n_head=n_head,
            vocab_size=self.args.vocab_size,
            block_size=self.args.block_size,
            
            # MLA parameters
            q_lora_rank=0,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            
            # MoE parameters
            use_moe=False,  # Set to False for dense model
            
            # RoPE parameters
            rope_theta=10000.0,
            
            # Precision
            fp8_params=getattr(self.args, 'use_fp8', False),
            fp8_mla_params=getattr(self.args, 'fp8_mla_params', False),
            
            # Other parameters
            dropout=self.args.dropout,
            bias=self.args.bias,
            attention_backend=getattr(self.args, 'attention_backend', None),
            use_gradient_checkpointing=True,
        )
        
        return config
        
    def _create_mla_model(self, config):
        """Create MLA-Model instance with the given configuration."""
        return MLAModel(config)

    def _create_gpt_config(self):
        from models.models.model import GPTConfig
        if self.args.size == 'small':
            config = GPTConfig(
                n_layer=8, n_head=8, n_embd=768, block_size=self.args.block_size,
                bias=self.args.bias, vocab_size=self.args.vocab_size, dropout=self.args.dropout,
                attention_backend=getattr(self.args, 'attention_backend', None)
            )
        elif self.args.size == 'medium':
            config = GPTConfig(
                n_layer=12, n_head=12, n_embd=1024, block_size=self.args.block_size,
                bias=self.args.bias, vocab_size=self.args.vocab_size, dropout=self.args.dropout,
                attention_backend=getattr(self.args, 'attention_backend', None)
            )
        else: # large
            config = GPTConfig(
                n_layer=24, n_head=16, n_embd=1536, block_size=self.args.block_size,
                bias=self.args.bias, vocab_size=self.args.vocab_size, dropout=self.args.dropout,
                attention_backend=getattr(self.args, 'attention_backend', None)
            )
        return config
    # --- End Config Creation Methods ---

    def forward(self, input_ids, targets=None, **kwargs):
        # Delegate forward pass to the underlying model
        # Handle different model signatures and potential extra outputs (like router_loss)
        model_type = self.args.model_type.lower()

        if model_type == 'llada':
            forward_args = {'input_ids': input_ids, 'targets': targets}
            if getattr(self.args, 'use_bd3_training', False):
                forward_args['use_bd3_training'] = True
            try:
                # LLaDA might return logits, loss, router_loss
                return self.model(**forward_args)
            except Exception as e:
                 print(f"Error during LLaDA forward pass: {e}")
                 if hasattr(self.model, 'forward_simple') and callable(getattr(self.model, 'forward_simple')):
                     logits, loss = self.model.forward_simple(input_ids, targets)
                     print("Used simplified forward pass to avoid NaN")
                     return logits, loss, None # Return None for router_loss
                 else:
                     raise
        elif model_type == 'mla':
            # MLA-Model returns logits, loss directly (no router loss in dense version)
            logits, loss = self.model(input_ids, targets)
            # No router loss in dense model
            return logits, loss, None
        else: # deepseek or gpt
            model_output = self.model(input_ids, targets=targets)

            if isinstance(model_output, tuple):
                if len(model_output) >= 2:
                    logits = model_output[0]
                    loss = model_output[1]
                    # Additional elements in model_output are ignored for this path.
                elif len(model_output) == 1:
                    logits = model_output[0]
                    loss = None # Loss will be calculated in training/validation step
                else: # Empty tuple
                    raise ValueError(
                        f"Model returned an empty tuple. Expected at least logits. Got: {model_output}"
                    )
            elif torch.is_tensor(model_output):
                # Model returned a single tensor, assume it's logits
                logits = model_output
                loss = None # Loss will be calculated in training/validation step
            else:
                raise ValueError(
                    f"Unexpected output type from model. Expected tensor or tuple, got: {type(model_output)}"
                )
            return logits, loss, None # Return None for router_loss consistency

    def training_step(self, batch, batch_idx):
        t0 = time.time()
        input_ids, targets = self._unpack_batch(batch)

        # Forward pass
        try:
            with self.timing_stats.track("forward"):
                # Completely detach and clone input tensors to prevent double backward issues
                with torch.no_grad():
                    input_ids = input_ids.detach().clone().requires_grad_(False)
                    targets = targets.detach().clone().requires_grad_(False)
                
                # Add special handling for MLA model
                if self.args.model_type.lower() == 'mla':
                    # Use a context manager from torch.autograd to prevent double backward
                    with torch.autograd.set_detect_anomaly(True):
                        logits, loss, router_loss = self(input_ids, targets=targets)
                else:
                    logits, loss, router_loss = self(input_ids, targets=targets)

            if loss is None: # Handle cases where loss is calculated outside model.forward
                # --- BEGIN SHAPE DIAGNOSTICS ---
                if batch_idx < 2 and self.global_rank == 0: # Log for first few batches on rank 0
                    print(f"TRAIN_STEP [{self.global_step}/{batch_idx}]: loss is None, calculating F.cross_entropy")
                    print(f"TRAIN_STEP [{self.global_step}/{batch_idx}]: logits original shape: {logits.shape if logits is not None else 'None'}")
                    print(f"TRAIN_STEP [{self.global_step}/{batch_idx}]: targets original shape: {targets.shape if targets is not None else 'None'}")
                    if logits is not None and targets is not None:
                        print(f"TRAIN_STEP [{self.global_step}/{batch_idx}]: logits.view(-1, logits.size(-1)) shape: {logits.view(-1, logits.size(-1)).shape}")
                        print(f"TRAIN_STEP [{self.global_step}/{batch_idx}]: targets.view(-1) shape: {targets.view(-1).shape}")
                # --- END SHAPE DIAGNOSTICS ---
                
                # Create fresh detached copies to avoid double backward
                logits_detached = logits.detach().clone().requires_grad_(True)
                loss = F.cross_entropy(logits_detached.view(-1, logits_detached.size(-1)), targets.view(-1), ignore_index=-1)

            # Handle router loss for MoE models
            if router_loss is not None and torch.is_tensor(router_loss):
                # Router loss should already be detached, but ensure it with torch.no_grad
                with torch.no_grad():
                    router_loss = router_loss.clone()
                
                # Check for NaN/inf in router_loss and handle
                if torch.isnan(router_loss).any() or torch.isinf(router_loss).any():
                    print(f"WARNING: NaN/Inf detected in router_loss at step {self.global_step}. Setting to zero.")
                    router_loss = torch.zeros_like(router_loss)

                # For MLA models, router loss is already incorporated internally
                model_type = getattr(self.args, 'model_type', 'gpt')
                
                if model_type == 'mla':
                    # Don't add router loss to prevent double counting
                    combined_loss = loss
                    # Still log the router loss for monitoring
                    if router_loss is not None:
                        self.log('train/router_loss', float(router_loss.item()), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
                else:
                    # For other models, add the router loss with coefficient
                    router_loss_coef = getattr(self.args, 'router_z_loss_coef', 0.001)
                    combined_loss = loss + router_loss_coef * router_loss
                    self.log('train/router_loss', float(router_loss.item()), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
            else:
                combined_loss = loss
                # Create a tensor with no gradient history
                with torch.no_grad():
                    router_loss = torch.tensor(0.0, device=self.device, requires_grad=False) # For logging consistency

            # Check for NaN/inf in combined_loss before logging/returning
            if torch.isnan(combined_loss).any() or torch.isinf(combined_loss).any():
                print(f"ERROR: NaN/Inf detected in combined_loss at step {self.global_step}. Using zero loss.")
                self.log('train/nan_loss_skipped', 1.0, on_step=True, on_epoch=False, sync_dist=True)
                # Create a new tensor with requires_grad=True to allow optimizer to run
                with torch.no_grad():
                    combined_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                    combined_loss = combined_loss.requires_grad_(True)


        except Exception as e:
            print(f"Error during training step {self.global_step}: {e}")
            print(traceback.format_exc())
            
            # Try to clean up memory to recover
            cleanup_memory()
            
            # Return a dummy loss to prevent crashing, log the error
            self.log('train/step_error', 1.0, on_step=True, on_epoch=False, sync_dist=True)
            
            # Create loss tensors with no gradient history
            combined_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32, requires_grad=True)
            with torch.no_grad():
                loss = torch.tensor(0.0, device=self.device, requires_grad=False)
                router_loss = torch.tensor(0.0, device=self.device, requires_grad=False)


        # --- Logging ---
        dt = time.time() - t0
        # Use float() to ensure no gradient tracking during logging
        self.log('train/loss', float(loss.item()), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        
        # Calculate gradient norms for monitoring
        if self.global_step % 100 == 0:  # Log grad norms less frequently
            with torch.no_grad():
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.log('train/grad_norm', total_norm, on_step=True, on_epoch=False, sync_dist=True)
        # Use float() for all tensor logging
        with torch.no_grad():
            self.log('train/combined_loss', float(combined_loss.item()), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log('train/step_time_ms', dt * 1000, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

        # Token/s calculation
        batch_tokens = input_ids.numel()
        self.total_tokens += batch_tokens * self.trainer.world_size # Accumulate across all devices
        self.tokens_window.append((time.time(), batch_tokens * self.trainer.world_size))
        if len(self.tokens_window) > self.window_size:
            self.tokens_window.pop(0)

        if len(self.tokens_window) > 1:
            window_time = self.tokens_window[-1][0] - self.tokens_window[0][0]
            window_tokens = sum(tokens for _, tokens in self.tokens_window)
            current_tokens_per_sec = window_tokens / window_time if window_time > 0 else 0
        else:
            current_tokens_per_sec = 0

        total_time_elapsed = time.time() - self.train_start_time
        avg_tokens_per_sec = self.total_tokens / total_time_elapsed if total_time_elapsed > 0 else 0

        self.log('tokens_per_sec_step', current_tokens_per_sec, on_step=True, on_epoch=False, prog_bar=True, sync_dist=False) # Log local Tps
        self.log('tokens_per_sec_avg', avg_tokens_per_sec, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log('total_tokens', float(self.total_tokens), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True) # Log as float for logger compatibility

        # MFU calculation (optional, requires estimate_mfu method on model)
        if hasattr(self.model, 'estimate_mfu') and self.trainer.global_step >= 5:
             # Estimate MFU based on total batch size across devices
             effective_batch_size = self.args.batch_size * self.trainer.accumulate_grad_batches * self.trainer.world_size
             mfu = self.model.estimate_mfu(effective_batch_size, dt) # dt is per-step time on this rank
             if mfu is not None:
                 self.running_mfu = mfu if self.running_mfu == -1.0 else 0.9 * self.running_mfu + 0.1 * mfu
                 self.log('perf/mfu_percent', self.running_mfu * 100, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        # Log timing stats periodically
        self.timing_stats.step()
        if self.timing_stats.should_print() and self.global_rank == 0:
            self.timing_stats.print_stats()
            # Log detailed timings if needed
            # for name, avg_time in self.timing_stats.get_average_times().items():
            #     self.log(f'timing/{name}_ms', avg_time * 1000, on_step=True, on_epoch=False, sync_dist=False)


        # Periodic memory cleanup and stats
        if self.global_step % 100 == 0:
            cleanup_memory()
            
        if self.global_rank == 0 and self.global_step % 1000 == 0:
            print_memory_stats(f"Step {self.global_step}")


        return combined_loss

    def validation_step(self, batch, batch_idx):
        input_ids, targets = self._unpack_batch(batch)

        # Forward pass
        logits, loss, router_loss = self(input_ids, targets=targets)

        if loss is None: # Handle cases where loss is calculated outside model.forward
            # --- BEGIN SHAPE DIAGNOSTICS ---
            if batch_idx < 2 and self.global_rank == 0: # Log for first few batches on rank 0
                print(f"VAL_STEP [{self.current_epoch}/{batch_idx}]: loss is None, calculating F.cross_entropy")
                print(f"VAL_STEP [{self.current_epoch}/{batch_idx}]: logits original shape: {logits.shape if logits is not None else 'None'}")
                print(f"VAL_STEP [{self.current_epoch}/{batch_idx}]: targets original shape: {targets.shape if targets is not None else 'None'}")
                if logits is not None and targets is not None:
                    print(f"VAL_STEP [{self.current_epoch}/{batch_idx}]: logits.view(-1, logits.size(-1)) shape: {logits.view(-1, logits.size(-1)).shape}")
                    print(f"VAL_STEP [{self.current_epoch}/{batch_idx}]: targets.view(-1) shape: {targets.view(-1).shape}")
            # --- END SHAPE DIAGNOSTICS ---
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        # Log validation loss
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Calculate and log perplexity
        perplexity = calculate_perplexity(loss)
        self.log('val/perplexity', perplexity, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        if router_loss is not None and torch.is_tensor(router_loss):
             self.log('val/router_loss', router_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss

    def configure_optimizers(self):
        """Sets up the optimizer and learning rate scheduler."""
        # Force AdamW for MLA models to avoid double backward issue with Lion
        if self.args.model_type == 'mla':
            print(f"Using AdamW optimizer for MLA model (avoids double backward)")
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                betas=(self.args.beta1, self.args.beta2)
            )
        elif hasattr(self.model, 'configure_optimizers') and self.args.optimizer_type is not None:
             optimizer = self.model.configure_optimizers(
                 weight_decay=self.args.weight_decay,
                 learning_rate=self.args.learning_rate, # Initial LR, scheduler will adjust
                 betas=(self.args.beta1, self.args.beta2),
                 device_type=self.device.type,
                 optimizer_type=getattr(self.args, 'optimizer_type', "AdamW") # Pass optimizer type if specified
             )
             print(f"Using optimizer configured by model: {type(optimizer)}")
        else:
             print(f"Using default AdamW optimizer")
             optimizer = AdamW(
                 self.model.parameters(),
                 lr=self.args.learning_rate,
                 weight_decay=self.args.weight_decay,
                 betas=(self.args.beta1, self.args.beta2)
             )

        # Learning rate scheduler
        if self.args.decay_lr:
            lr_scheduler = {
                'scheduler': LambdaLR(optimizer, lr_lambda=self._lr_lambda),
                'interval': 'step', # Call scheduler every step
                'frequency': 1,
                'name': 'learning_rate_scheduler'
            }
            print("Using learning rate decay scheduler.")
            return [optimizer], [lr_scheduler]
        else:
            print("Using constant learning rate.")
            return optimizer

    def _lr_lambda(self, current_step: int):
        """Lambda function for LR scheduler based on original get_lr logic."""
        # Adjust step by 1 because PL scheduler steps *before* optimizer step
        # However, PL logs LR *after* the step, so using current_step directly might be correct
        # Let's stick to the original logic's step counting if possible.
        # PL's global_step should align with iter_num if accumulation=1
        # If using gradient accumulation, PL's global_step increments every optimizer step,
        # which matches the intent of iter_num in the original code.
        iter_num = self.global_step + 1 # global_step starts at 0

        return get_lr(
            current_iter=iter_num,
            warmup_iters=self.args.warmup_iters,
            lr_decay_iters=self.args.lr_decay_iters,
            learning_rate=self.args.learning_rate, # Base LR
            min_lr=self.args.min_lr,
        ) / self.args.learning_rate # Lambda returns a multiplicative factor


    def on_validation_epoch_end(self):
        """Called at the end of the validation epoch."""
        # Generate sample text only on rank 0
        if self.global_rank == 0 and hasattr(self.args, 'tokenizer') and self.args.tokenizer:
            self.generate_sample_text()

        # Optional: Perform more complex validation loss estimation like in original code
        # This might involve running estimate_loss utility if needed, but PL's logging
        # over the validation dataloader should be sufficient.
        # losses = estimate_loss(...) # If needed
        # self.log('val/estimated_loss', losses['val'], sync_dist=True)
        # self.log('val/estimated_perplexity', losses['val_ppl'], sync_dist=True)
        pass


    def generate_sample_text(self):
        """Generates sample text using the current model."""
        print("\n--- Generating Sample Text ---")
        if not hasattr(self.args, 'tokenizer') or not self.args.tokenizer:
            print("Tokenizer not available, skipping text generation.")
            return

        tokenizer = self.args.tokenizer
        prompt = self._get_random_prompt()
        print(f"Prompt: {prompt}")

        try:
            input_tokens = tokenizer.encode(
                prompt, return_tensors='pt', add_special_tokens=True
            ).to(self.device)

            # Ensure model is in eval mode for generation
            self.model.eval()

            with torch.no_grad(): # No need for gradients during generation
                 # Use the generic generate_text utility function
                 output_text = generate_text(
                     self.model, # Pass the LightningModule's model
                     input_tokens,
                     max_new_tokens=min(100, self.args.block_size - input_tokens.shape[1]),
                     temperature=0.7,
                     top_k=40,
                     tokenizer=tokenizer
                 )

            # Switch back to train mode
            self.model.train()

            if output_text:
                print(f"Generated: {output_text}")
            else:
                print("Generation finished (output format depends on model type or failed).")

        except Exception as e:
            print(f"Error during text generation: {e}")
            print(traceback.format_exc())
            # Ensure model is back in train mode even if generation fails
            self.model.train()
        print("--- End Sample Text ---\n")


    def _get_random_prompt(self):
        """Selects a random prompt for generation."""
        if hasattr(self.args, 'prompt_templates') and self.args.prompt_templates:
            return random.choice(self.args.prompt_templates)
        else:
            diverse_prompts = [
                "Once upon a time", "In a distant world", "The story begins with",
                "How can one solve", "Why are humans", "What is the best way to",
                "Explain to me how", "Write a guide for",
                "Imagine a scenario where", "Describe a futuristic technology that",
                "Analyze the advantages and disadvantages of",
                "Compare and contrast the following approaches:"
            ]
            return random.choice(diverse_prompts)

    def _unpack_batch(self, batch):
        """Unpacks batch into input_ids and targets, handling different formats."""
        if isinstance(batch, dict):
            input_ids = batch['input_ids']
            targets = batch.get('labels', input_ids) # Use input_ids as targets if labels missing
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            input_ids = batch[0]
            targets = batch[-1] # Assume last element is target
        elif torch.is_tensor(batch):
             input_ids = batch
             targets = batch # Use input_ids as targets
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

        # Ensure targets have the same shape as input_ids if needed by loss function
        # Often, targets are shifted inside the model's forward pass or loss calculation
        # Return them as they are from the dataloader for flexibility.
        return input_ids, targets

    # Optional: Add hooks for setup, cleanup, etc. if needed
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
             # Code to run before training starts (e.g., print info)
             if self.global_rank == 0:
                 print("Starting training setup...")
                 print(f"World size: {self.trainer.world_size}")
                 print(f"Global batch size: {self.args.batch_size * self.trainer.world_size * self.trainer.accumulate_grad_batches}")
                 print(f"Gradient accumulation steps: {self.trainer.accumulate_grad_batches}")
                 print(f"Using precision: {self.trainer.precision}")


    def teardown(self, stage=None):
         if stage == 'fit' or stage is None:
             # Code to run after training finishes
             cleanup_memory()
             if self.global_rank == 0:
                 print("Training finished. Final memory stats:")
                 print_memory_stats("Teardown")


# Note: DataModule definition would go here or in a separate file.
# For now, we assume dataloaders are passed directly to trainer.fit()