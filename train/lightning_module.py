import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import os
import time
import traceback
import random
import csv
import shutil
from datetime import datetime
import wandb

# Import necessary components from your project
from models.deepseek.deepseek_adapter_mtp import DeepSeekMiniMTP
from models.llada.model import LLaDAModel
from models.models.model import GPT
from models.sedd.model import SEDDModel
from train.model_configs import *
from train.train_utils import (
    get_lr, calculate_perplexity, ensure_model_dtype,
    AveragedTimingStats, generate_text, estimate_loss
)
from optimization.memory_optim import cleanup_memory, print_memory_stats, preallocate_cuda_memory

try:
    from torch.utils.checkpoint import _StopRecomputationError as _CheckpointStop
except ImportError:  # Older torch versions may not expose this symbol
    _CheckpointStop = None
from optimization.cuda_optim import setup_cuda_optimizations, print_gpu_stats
from optimization.training_optim import enable_torch_compile, autoconfigure_environment, optimize_attention_operations
from optimization.fp8_deepseek_trainer import FP8AdamW, FP8MixedPrecisionTrainer
from models.galore2_fixed import GaLore2AdamW


class LLMLightningModule(pl.LightningModule):
    """PyTorch Lightning Module for training LLMs."""

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args) # Saves args to self.hparams
        self.args = args # Keep args accessible directly too
        
        # Block size adaptation parameters
        self.load_checkpoint_path = getattr(args, 'load_checkpoint_path', None)
        self.original_block_size = getattr(args, 'original_block_size', None)
        
        # Progressive training parameters
        self.progressive_training = getattr(args, 'progressive_training', False)
        self.progressive_block_sizes = getattr(args, 'progressive_block_sizes', [512, 1024, 2048, 4096])
        self.progressive_epochs_per_stage = getattr(args, 'progressive_epochs_per_stage', 5)
        self.progressive_current_stage = 0
        self.progressive_lr_scale = getattr(args, 'progressive_lr_scale', 0.5)  # LR reduction at transitions
        
        # Initialize wandb if enabled
        self.use_wandb = getattr(args, 'use_wandb', True)
        if self.use_wandb and self.global_rank == 0:
            self._init_wandb()
        
        # Early environment configuration for optimal kernels and allocator
        try:
            autoconfigure_environment()
        except Exception:
            pass
        self.model = self._build_model()

        # Handle checkpoint loading with block size adaptation if needed
        if self.load_checkpoint_path and self.original_block_size:
            self._load_and_adapt_checkpoint()

        # CRITICAL FIX: Force model to CPU before DDP wrapping
        # This prevents rank 1 from having duplicated memory (~17GB extra)
        # Lightning/DDP will properly move it to correct GPUs
        print(f"Moving model to CPU before DDP wrapping to avoid VRAM imbalance...")
        self.model = self.model.cpu()

        # Also move large buffers to CPU
        if hasattr(self.model, 'freqs_cis') and self.model.freqs_cis is not None:
            self.model.freqs_cis = self.model.freqs_cis.cpu()
            
        self.timing_stats = AveragedTimingStats(print_interval=1000)
        self.train_start_time = time.time()
        self.total_tokens = 0
        self.tokens_window = []
        self.window_size = 10
        self.running_mfu = -1.0
        
        # CSV logging attributes
        self.metrics_buffer = []  # Store all metrics for each step
        self.csv_file_path = None
        self.csv_writer = None
        self.csv_file = None
        # Keep track of last validation metrics
        self.last_val_loss = None
        self.last_val_perplexity = None
        
        # Apply CUDA optimizations if available and requested
        if torch.cuda.is_available():
            setup_cuda_optimizations()
            # Enable optimized attention kernels (PyTorch SDPA/Flash SDP) when requested
            # if getattr(self.args, 'optimize_attention', True):
            #     try:
            #         optimize_attention_operations()
            #     except Exception:
            #         pass
            if hasattr(self.args, 'preallocate_memory') and self.args.preallocate_memory:
                preallocate_cuda_memory()
            if self.global_rank == 0:
                 print_gpu_stats()
        
        # Compile model if requested
        if hasattr(self.args, 'compile') and self.args.compile:
            # Configure torch._dynamo for better compatibility with dynamic shapes
            # import torch._dynamo as dynamo
            # # Increase cache size limit to handle more shape variations
            # dynamo.config.cache_size_limit = 64  # Default is 8
            # # Continue on errors instead of crashing
            # dynamo.config.suppress_errors = True
            # print(f"Configured torch._dynamo with cache_size_limit=64 for dynamic batching")
            
            # Skip compilation for models with known torch.compile compatibility issues
            skip_compile_models = ['mla_llada']
            if self.args.model_type.lower() in skip_compile_models:
                print(f"Skipping model compilation for {self.args.model_type} model (torch.compile compatibility issues with gradient checkpointing)")
            else:
                disabled_ckpt = self._disable_gradient_checkpointing_for_compile()
                if disabled_ckpt:
                    print(f"Disabled gradient checkpointing for {disabled_ckpt} modules prior to torch.compile")
                # For MLA-selective model, disable gradient checkpointing before compilation
                if self.args.model_type.lower() == 'mla_selective':
                    print("Disabling gradient checkpointing for MLA-selective model compilation...")
                    # Already handled in _disable_gradient_checkpointing_for_compile, keep message for clarity
                
                print("Compiling model with torch.compile...")
                try:
                    # Use reduce-overhead mode for models with complex attention patterns
                    compile_mode = 'max-autotune'
                    self.model = enable_torch_compile(
                        self.model,
                        mode=compile_mode,
                        backend='inductor'
                    )
                    print(f"Model compilation successful with mode: {compile_mode}!")
                except Exception as e:
                    print(f"Model compilation failed: {e}")
                    print("Continuing without compilation...")

        # Print model size
        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
        print(f"Initialized {self.args.model_type} model ({self.args.size}) with {param_count:.2f}M parameters.")
        print(f"Trainable parameters: {trainable_count:.2f}M")
        
        # Special logging for ParScale stage 2
        if self.args.model_type == 'parscale_mla' and getattr(self.args, 'training_stage', 1) == 2:
            print(f"ParScale Stage 2: Training only ParScale components ({trainable_count:.2f}M params)")
            print(f"Base model parameters frozen: {(param_count - trainable_count):.2f}M params")
            
        print_memory_stats("After Model Init")
        
        # Initialize CSV logging
        self._init_csv_logging()
        
        # Watch model with wandb if enabled
        if self.use_wandb and self.global_rank == 0:
            try:
                wandb.watch(self.model, log=None, log_freq=100)
            except Exception as e:
                print(f"Warning: Failed to watch model with wandb: {e}")

    def _disable_gradient_checkpointing_for_compile(self):
        """Disable checkpointing hooks that are incompatible with torch.compile."""
        disabled = 0
        if not hasattr(self, 'model'):
            return disabled

        for module in self.model.modules():
            if hasattr(module, 'use_checkpoint') and getattr(module, 'use_checkpoint', False):
                module.use_checkpoint = False
                disabled += 1
            if hasattr(module, 'gradient_checkpointing') and getattr(module, 'gradient_checkpointing', False):
                try:
                    module.gradient_checkpointing = False
                except Exception:
                    pass
            if hasattr(module, 'gradient_checkpointing_enable') and callable(module.gradient_checkpointing_enable):
                try:
                    module.gradient_checkpointing_enable(False)
                except Exception:
                    pass

        for attr in ('use_gradient_checkpointing', 'gradient_checkpointing'):
            if hasattr(self.args, attr):
                setattr(self.args, attr, False)

        return disabled

    def _init_wandb(self):
        """Initialize wandb logging with project configuration."""
        try:
            wandb.login(key=os.getenv('WANDB_API_KEY'))
            # Create wandb config from args
            wandb_config = {
                'model_type': self.args.model_type,
                'size': self.args.size,
                'batch_size': self.args.batch_size,
                'block_size': self.args.block_size,
                'learning_rate': self.args.learning_rate,
                'dropout': getattr(self.args, 'dropout', 0.1),
                'vocab_size': self.args.vocab_size,
                'weight_decay': getattr(self.args, 'weight_decay', 0.01),
                'warmup_iters': getattr(self.args, 'warmup_iters', 2000),
                'lr_decay_iters': getattr(self.args, 'lr_decay_iters', 600000),
                'min_lr': getattr(self.args, 'min_lr', 6e-5),
                'beta1': getattr(self.args, 'beta1', 0.9),
                'beta2': getattr(self.args, 'beta2', 0.95),
                'grad_clip': getattr(self.args, 'grad_clip', 1.0),
                'compile': getattr(self.args, 'compile', False),
                'use_fp8': getattr(self.args, 'use_fp8', False),
                'optimizer_type': getattr(self.args, 'optimizer_type', 'adamw'),
                'dataset': getattr(self.args, 'dataset', 'apollo-mini'),
            }
            
            # Add model-specific configs
            if hasattr(self.args, 'num_experts'):
                wandb_config['num_experts'] = self.args.num_experts
            if hasattr(self.args, 'experts_per_token'):
                wandb_config['experts_per_token'] = self.args.experts_per_token
            if hasattr(self.args, 'shared_weight_ratio'):
                wandb_config['shared_weight_ratio'] = self.args.shared_weight_ratio
                
            # Initialize wandb
            wandb.init(
                project=getattr(self.args, 'wandb_project', 'gptoughts-training'),
                name=getattr(self.args, 'wandb_run_name', f"{self.args.model_type}_{self.args.size}"),
                config=wandb_config,
                tags=[self.args.model_type, self.args.size],
                resume=getattr(self.args, 'wandb_resume', None)
            )
            print("Wandb initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            self.use_wandb = False

    def _build_model(self):
        """Initializes the model based on configuration."""
        model_type = self.args.model_type.lower()
        print(f"Building model: {model_type} size: {self.args.size}")

        # Determine config based on model type and size
        if model_type == 'deepseek':
            config = create_deepseek_config(self.args)
            model = DeepSeekMiniMTP(config)
        elif model_type == 'llada':
            config = create_llada_config(self.args)
            model = LLaDAModel(config)
        elif model_type == 'sedd':
            config = create_sedd_config(self.args)
            model = SEDDModel(config)
        elif model_type == 'mla':
            config = create_mla_config(self.args)
            model = create_mla_model(config)
        elif model_type == 'mla_selective':
            config = create_mla_selective_config(self.args)
            model = create_mla_selective_model(config)
        elif model_type == 'parscale_mla':
            config = create_parscale_mla_config(self.args)
            model = create_parscale_mla_model(config)
        elif model_type == 'mla_llada':
            config = create_mla_llada_config(self.args)
            model = create_mla_llada_model(config)
        elif model_type == 'mdm':
            config = create_mdm_config(self.args)
            model = create_mdm_model(config)
        elif model_type == 'moe_mla':
            config = create_moe_mla_config(self.args)
            model = create_moe_mla_model(config)
        elif model_type == 'slm':
            config = create_slm_config(self.args)
            model = create_slm_model(config)
        elif model_type == 'nsa':
            config = create_nsa_config(self.args)
            model = create_nsa_model(config)
        elif model_type == 'hse':
            config = create_hse_config(self.args)
            model = create_hse_model(config)
        elif model_type == 'swan':
            config = create_swan_config(self.args)
            from models.models.swan_model import SWANModel
            model = SWANModel(config)
        elif model_type == 'swa_mla':
            config = create_swa_mla_config(self.args)
            from models.models.swa_mla_model import SWAMLAModel
            model = SWAMLAModel(config)
        elif model_type == 'swa_mla_moe':
            config = create_swa_mla_moe_config(self.args)
            from models.models.swa_mla_moe_model import SWAMLAMOEModel
            model = SWAMLAMOEModel(config)
        elif model_type == 'hrm':
            config = create_hrm_config(self.args)
            model = create_hrm_model(config)
        elif model_type == 'adaptive_moe':
            from train.model_configs.adaptive_moe_config import create_adaptive_moe_config, create_adaptive_moe_model
            config = create_adaptive_moe_config(self.args)
            model = create_adaptive_moe_model(config)
        else: # gpt
            config = create_gpt_config(self.args)
            model = GPT(config)

        self.config = config # Store config for potential use later
        return model

    
    def adapt_to_new_block_size(self, model, old_block_size, new_block_size):
        """
        Adapts a model trained on old_block_size to work with new_block_size.
        Handles different types of position encoding (fixed embeddings, RoPE, etc.)
        """
        if old_block_size == new_block_size:
            print(f"Block size unchanged ({new_block_size}), no adaptation needed")
            return model
            
        print(f"Adapting model from block_size {old_block_size} to {new_block_size}")
        model_type = self.args.model_type.lower()
        
        if model_type in ['gpt', 'mdm']:
            self._adapt_fixed_position_embeddings(model, old_block_size, new_block_size)
        elif model_type in ['mla', 'mla_selective', 'parscale_mla', 'moe_mla', 'slm', 'nsa', 'adaptive_moe']:
            self._adapt_rope_position_encoding(model, old_block_size, new_block_size)
        elif model_type == 'llada':
            self._adapt_llada_position_encoding(model, old_block_size, new_block_size)
        elif model_type in ['deepseek']:
            self._adapt_deepseek_position_encoding(model, old_block_size, new_block_size)
        elif model_type == 'sedd':
            self._adapt_sedd_position_encoding(model, old_block_size, new_block_size)
        else:
            print(f"Warning: No specific adaptation implemented for model type {model_type}")
            
        return model
    
    def _adapt_fixed_position_embeddings(self, model, old_size, new_size):
        """Adapt models with fixed position embeddings (nn.Embedding)."""
        print(f"Adapting fixed position embeddings from {old_size} to {new_size}")
        
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'wpe'):
            old_pos_embed = model.transformer.wpe
            old_weights = old_pos_embed.weight.data.clone()
            
            # Create new position embedding layer
            new_pos_embed = torch.nn.Embedding(new_size, old_pos_embed.embedding_dim)
            
            if new_size > old_size:
                # Extend: copy old weights and interpolate new positions
                new_pos_embed.weight.data[:old_size] = old_weights
                
                # Interpolate remaining positions
                for i in range(old_size, new_size):
                    # Linear interpolation from the last few positions
                    if old_size >= 4:
                        # Use weighted average of last 4 positions
                        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
                        interpolated = torch.sum(old_weights[-4:] * weights.unsqueeze(1), dim=0)
                    else:
                        # If too few positions, just use the last one with noise
                        interpolated = old_weights[-1] + torch.randn_like(old_weights[-1]) * 0.02
                    
                    new_pos_embed.weight.data[i] = interpolated
                    
            else:
                # Truncate: just take the first new_size positions
                new_pos_embed.weight.data = old_weights[:new_size]
            
            # Replace the old embedding
            model.transformer.wpe = new_pos_embed
            print(f"Position embeddings adapted: {old_size} -> {new_size}")
            
        elif hasattr(model, 'position_embedding'):
            # For models like MDM
            old_pos_embed = model.position_embedding
            old_weights = old_pos_embed.weight.data.clone()
            
            new_pos_embed = torch.nn.Embedding(new_size, old_pos_embed.embedding_dim)
            
            if new_size > old_size:
                new_pos_embed.weight.data[:old_size] = old_weights
                # Interpolate remaining positions
                for i in range(old_size, new_size):
                    if old_size >= 4:
                        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
                        interpolated = torch.sum(old_weights[-4:] * weights.unsqueeze(1), dim=0)
                    else:
                        interpolated = old_weights[-1] + torch.randn_like(old_weights[-1]) * 0.02
                    new_pos_embed.weight.data[i] = interpolated
            else:
                new_pos_embed.weight.data = old_weights[:new_size]
                
            model.position_embedding = new_pos_embed
            print(f"Position embeddings adapted: {old_size} -> {new_size}")
    
    def _adapt_rope_position_encoding(self, model, old_size, new_size):
        """Adapt models with RoPE position encoding."""
        print(f"Adapting RoPE encoding from {old_size} to {new_size}")
        
        # For RoPE, we need to extend the cached cos/sin values
        def extend_rope_cache(module):
            if hasattr(module, 'rope'):
                rope = module.rope
                if hasattr(rope, '_extend_cos_sin_cache'):
                    rope._extend_cos_sin_cache(new_size)
                    print(f"Extended RoPE cache in {module.__class__.__name__} to {new_size}")
                elif hasattr(rope, 'max_seq_len') and rope.max_seq_len < new_size:
                    # Recreate RoPE with new max_seq_len
                    from models.blocks.positional_encoding import RoPE
                    new_rope = RoPE(rope.dim, max_seq_len=new_size, base=rope.base)
                    module.rope = new_rope
                    print(f"Recreated RoPE in {module.__class__.__name__} with max_seq_len={new_size}")
        
        # Recursively find and update RoPE modules
        def update_rope_recursive(module):
            extend_rope_cache(module)
            for child in module.children():
                update_rope_recursive(child)
        
        update_rope_recursive(model)
        
        # Apply position interpolation for better generalization
        if new_size > old_size and hasattr(self.args, 'use_position_interpolation') and self.args.use_position_interpolation:
            self._apply_position_interpolation(model, old_size, new_size)
    
    def _adapt_llada_position_encoding(self, model, old_size, new_size):
        """Adapt LLaDA model position encoding."""
        print(f"Adapting LLaDA position encoding from {old_size} to {new_size}")
        
        # LLaDA uses RoPE in attention layers
        self._adapt_rope_position_encoding(model, old_size, new_size)
        
        # Update BD3 block length if needed
        if hasattr(model, 'config'):
            if hasattr(model.config, 'bd3_block_length'):
                # Scale BD3 block length proportionally
                old_bd3_block = getattr(model.config, 'bd3_block_length', old_size // 4)
                new_bd3_block = int(old_bd3_block * new_size / old_size)
                model.config.bd3_block_length = max(32, new_bd3_block)  # Minimum sensible block size
                print(f"Updated BD3 block length: {old_bd3_block} -> {model.config.bd3_block_length}")
    
    def _adapt_deepseek_position_encoding(self, model, old_size, new_size):
        """Adapt DeepSeek model position encoding."""
        print(f"Adapting DeepSeek position encoding from {old_size} to {new_size}")
        
        # DeepSeek uses RoPE
        self._adapt_rope_position_encoding(model, old_size, new_size)
        
        # Update max_position_embeddings in config if present
        if hasattr(model, 'config') and hasattr(model.config, 'max_position_embeddings'):
            model.config.max_position_embeddings = new_size
            print(f"Updated max_position_embeddings to {new_size}")
    
    def _adapt_sedd_position_encoding(self, model, old_size, new_size):
        """Adapt SEDD model position encoding."""
        print(f"Adapting SEDD position encoding from {old_size} to {new_size}")
        
        # SEDD uses RoPE in transformer blocks
        self._adapt_rope_position_encoding(model, old_size, new_size)
    
    def _apply_position_interpolation(self, model, old_size, new_size):
        """
        Apply Position Interpolation (PI) to RoPE frequencies.
        Scales the frequencies to maintain learned positional relationships.
        """
        print(f"Applying position interpolation: scaling factor = {new_size/old_size:.2f}")
        
        def interpolate_rope_freqs(module):
            if hasattr(module, 'rope'):
                rope = module.rope
                if hasattr(rope, 'base'):
                    # Calculate new base frequency
                    scale_factor = new_size / old_size
                    new_base = rope.base * (scale_factor ** (rope.dim / (rope.dim - 2)))
                    
                    # Recreate RoPE with interpolated frequencies
                    from models.blocks.positional_encoding import RoPE
                    new_rope = RoPE(rope.dim, max_seq_len=new_size, base=new_base)
                    module.rope = new_rope
                    print(f"Applied PI to {module.__class__.__name__}: base {rope.base:.0f} -> {new_base:.0f}")
        
        def apply_pi_recursive(module):
            interpolate_rope_freqs(module)
            for child in module.children():
                apply_pi_recursive(child)
        
        apply_pi_recursive(model)
    
    def _clean_checkpoint_keys(self, state_dict):
        """
        Clean checkpoint keys to handle different formats:
        - Remove _orig_mod prefix from torch.compile()
        - Remove model. prefix from Lightning checkpoints
        - Handle other common prefixes
        """
        cleaned_state_dict = {}
        
        # Analyze the key patterns to choose the best cleaning strategy
        sample_keys = list(state_dict.keys())[:10]
        print(f"Sample checkpoint keys: {sample_keys}")
        
        # Detect the pattern
        has_model_prefix = any(key.startswith('model.') for key in sample_keys)
        has_orig_mod_prefix = any('_orig_mod.' in key for key in sample_keys)
        
        print(f"Key pattern analysis: model.={has_model_prefix}, _orig_mod.={has_orig_mod_prefix}")
        
        for key, value in state_dict.items():
            clean_key = key
            
            # Apply cleaning in order based on detected patterns
            if has_orig_mod_prefix and '_orig_mod.' in clean_key:
                # Handle model._orig_mod.xxx or _orig_mod.xxx
                if clean_key.startswith('model._orig_mod.'):
                    clean_key = clean_key.replace('model._orig_mod.', '')
                elif clean_key.startswith('_orig_mod.'):
                    clean_key = clean_key.replace('_orig_mod.', '')
            elif has_model_prefix and clean_key.startswith('model.'):
                # Simple Lightning checkpoint format
                clean_key = clean_key.replace('model.', '')
            
            # Additional prefixes
            if clean_key.startswith('_forward_module.'):
                clean_key = clean_key.replace('_forward_module.', '')
                
            cleaned_state_dict[clean_key] = value
            
        return cleaned_state_dict
    
    def _is_model_compiled(self, state_dict):
        """Check if the model state dict is from a compiled model."""
        return any(key.startswith('_orig_mod.') for key in state_dict.keys())
    
    def _add_compile_prefix_to_state_dict(self, state_dict):
        """Add _orig_mod prefix to all keys in state dict for compiled models."""
        compiled_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                # Replace model. with model._orig_mod.
                new_key = key.replace('model.', 'model._orig_mod.')
            elif not key.startswith('_orig_mod.'):
                # Add _orig_mod prefix if not already present
                new_key = f'_orig_mod.{key}' if not key.startswith('model.') else key.replace('model.', 'model._orig_mod.')
            else:
                new_key = key
            compiled_state_dict[new_key] = value
        return compiled_state_dict
    
    def _load_and_adapt_checkpoint(self):
        """Load checkpoint and adapt model to new block size if needed."""
        print(f"Loading checkpoint from: {self.load_checkpoint_path}")
        
        try:
            # Load checkpoint (weights_only=False for PyTorch 2.6+ compatibility)
            checkpoint = torch.load(self.load_checkpoint_path, map_location='cpu', weights_only=False)
            
            # Extract model state dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Clean checkpoint keys (handle _orig_mod, model. prefixes, etc.)
            original_keys = len(state_dict)
            state_dict = self._clean_checkpoint_keys(state_dict)
            
            # Check what prefixes were cleaned
            compiled_keys = [k for k in checkpoint.get('state_dict', checkpoint).keys() if '_orig_mod.' in k]
            lightning_keys = [k for k in checkpoint.get('state_dict', checkpoint).keys() if k.startswith('model.') and '_orig_mod' not in k]
            
            if compiled_keys:
                print(f"Detected compiled model checkpoint ({len(compiled_keys)} keys with _orig_mod prefix)")
            if lightning_keys:
                print(f"Detected Lightning checkpoint format ({len(lightning_keys)} keys with model. prefix)")
                
            print(f"Cleaned {original_keys} checkpoint keys")
            
            # Load state dict into model (strict=False to handle potential size mismatches)
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            # Check if loading was successful
            total_model_params = len([name for name, _ in self.model.named_parameters()])
            loaded_params = len(state_dict) - len(missing_keys)
            loading_success_rate = loaded_params / len(state_dict) * 100 if len(state_dict) > 0 else 0
            
            print(f"=== Checkpoint Loading Statistics ===")
            print(f"Total parameters in checkpoint: {len(state_dict)}")
            print(f"Total parameters in model: {total_model_params}")
            print(f"Successfully loaded: {loaded_params} ({loading_success_rate:.1f}%)")
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")
            
            if missing_keys:
                print(f"First 5 missing keys: {missing_keys[:5]}")
            if unexpected_keys:
                print(f"First 5 unexpected keys: {unexpected_keys[:5]}")
                
            # Critical validation: if too many keys are missing, abort
            if len(missing_keys) > len(state_dict) * 0.8:  # More than 80% missing
                error_msg = f"CRITICAL ERROR: {len(missing_keys)} out of {len(state_dict)} keys missing ({100 - loading_success_rate:.1f}% failure rate). This indicates the model weights were not loaded properly."
                print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)
            elif len(missing_keys) > len(state_dict) * 0.5:  # More than 50% missing
                print(f"WARNING: {len(missing_keys)} keys missing ({100 - loading_success_rate:.1f}% failure rate). Model may not work properly.")
                
            print(f"Checkpoint loaded with {loading_success_rate:.1f}% success rate")
            
            # Adapt to new block size if needed
            if self.original_block_size != self.args.block_size:
                print(f"Adapting model from block_size {self.original_block_size} to {self.args.block_size}")
                
                # Store some weights before adaptation to verify they're preserved
                sample_weights_before = {}
                for name, param in self.model.named_parameters():
                    if 'transformer.wte.weight' in name or 'transformer.h.0.norm1.weight' in name:
                        sample_weights_before[name] = param.data.clone()
                        print(f"Sample weight before adaptation - {name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}")
                        break
                
                self.model = self.adapt_to_new_block_size(self.model, self.original_block_size, self.args.block_size)
                
                # Check that main weights are preserved after adaptation
                for name, param in self.model.named_parameters():
                    if name in sample_weights_before:
                        weight_diff = (param.data - sample_weights_before[name]).abs().mean()
                        print(f"Sample weight after adaptation - {name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}, diff={weight_diff:.8f}")
                        if weight_diff > 1e-6:
                            print(f"WARNING: Weight {name} changed significantly during adaptation!")
                        else:
                            print(f"✓ Weight {name} preserved correctly during adaptation")
                        break
                
                print("Model adaptation completed")
                
                # Update config to reflect new block size
                if hasattr(self.model, 'config'):
                    self.model.config.block_size = self.args.block_size
                if hasattr(self, 'config'):
                    self.config.block_size = self.args.block_size
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print(traceback.format_exc())
            raise e
    
    def setup_progressive_training(self):
        """Setup progressive training if enabled."""
        if not self.progressive_training:
            return
            
        # Start with the smallest block size
        if len(self.progressive_block_sizes) > 0:
            initial_block_size = self.progressive_block_sizes[0]
            if initial_block_size != self.args.block_size:
                print(f"Progressive training: starting with block_size={initial_block_size}")
                old_block_size = self.args.block_size
                self.args.block_size = initial_block_size
                
                # Adapt model if needed
                if hasattr(self.model, 'config'):
                    self.model.config.block_size = initial_block_size
                if hasattr(self, 'config'):
                    self.config.block_size = initial_block_size
                    
                # Adapt the model to the smaller size if coming from larger
                if old_block_size > initial_block_size:
                    self.model = self.adapt_to_new_block_size(self.model, old_block_size, initial_block_size)
    
    def should_progress_to_next_stage(self):
        """Check if we should move to the next progressive training stage."""
        if not self.progressive_training:
            return False
            
        # Check if we've completed enough epochs for current stage
        epochs_in_stage = self.current_epoch - (self.progressive_current_stage * self.progressive_epochs_per_stage)
        
        return (epochs_in_stage >= self.progressive_epochs_per_stage and 
                self.progressive_current_stage < len(self.progressive_block_sizes) - 1)
    
    def progress_to_next_stage(self):
        """Progress to next stage in progressive training."""
        if not self.progressive_training or self.progressive_current_stage >= len(self.progressive_block_sizes) - 1:
            return False
            
        old_stage = self.progressive_current_stage
        old_block_size = self.progressive_block_sizes[old_stage]
        
        self.progressive_current_stage += 1
        new_block_size = self.progressive_block_sizes[self.progressive_current_stage]
        
        print(f"\n=== Progressive Training: Stage {old_stage} -> {self.progressive_current_stage} ===")
        print(f"Transitioning from block_size {old_block_size} to {new_block_size}")
        
        # Update args and configs
        self.args.block_size = new_block_size
        if hasattr(self.model, 'config'):
            self.model.config.block_size = new_block_size
        if hasattr(self, 'config'):
            self.config.block_size = new_block_size
        
        # Adapt model to new block size
        self.model = self.adapt_to_new_block_size(self.model, old_block_size, new_block_size)
        
        # Scale learning rate down for stability
        if hasattr(self.trainer, 'optimizers') and len(self.trainer.optimizers) > 0:
            optimizer = self.trainer.optimizers[0]
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] = old_lr * self.progressive_lr_scale
                print(f"Scaled learning rate: {old_lr:.2e} -> {param_group['lr']:.2e}")
        
        print(f"=== Stage transition completed ===\n")
        
        # Log to wandb if available
        if self.use_wandb and self.global_rank == 0:
            try:
                wandb.log({
                    'progressive_training/stage': self.progressive_current_stage,
                    'progressive_training/block_size': new_block_size,
                    'progressive_training/transition_epoch': self.current_epoch
                }, step=self.global_step)
            except Exception as e:
                print(f"Warning: Failed to log progressive training to wandb: {e}")
        
        return True
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        # Check for progressive training transitions
        if self.should_progress_to_next_stage():
            self.progress_to_next_stage()
            
            # Force a checkpoint save after stage transition
            if self.global_rank == 0 and hasattr(self.trainer, 'save_checkpoint'):
                checkpoint_path = f"progressive_stage_{self.progressive_current_stage}_epoch_{self.current_epoch}.ckpt"
                self.trainer.save_checkpoint(checkpoint_path)
                print(f"Saved checkpoint after stage transition: {checkpoint_path}")

    def forward(self, input_ids, targets=None, **kwargs):
        """
        Unified forward pass that harmonizes outputs from different models
        into a standard dictionary format: {'logits': ..., 'loss': ..., 'router_loss': ...}.
        """
        model_type = self.args.model_type.lower()
        
        # Prepare keyword arguments, separating the positional `input_ids`.
        model_kwargs = {}
        if targets is not None:
            # Handle model-specific names for the target/label tensor.
            if model_type in ['mla_llada', 'mdm', 'hrm', 'sedd']:
                model_kwargs['labels'] = targets
            else:
                model_kwargs['targets'] = targets

        # Add any other keyword arguments passed to this forward method.
        model_kwargs.update(kwargs)

        # Do not force BD3 here; training_step passes it explicitly for train only

        # Call the model. By passing `input_ids` as the first positional argument,
        # we support models that name it `idx` (like NSA) or `input_ids`.
        # The remaining arguments are passed by keyword.
        model_output = self.model(input_ids, **model_kwargs)
        
        # --- Harmonize Output ---
        logits, loss, router_loss = None, None, None

        if isinstance(model_output, dict):
            # Models that return dictionaries (e.g., MLA-LLaDA, MDM, HRM)
            logits = model_output.get('logits')
            loss = model_output.get('loss')
            router_loss = model_output.get('router_loss') # Will be None if not present
            
            # Handle HRM's special losses
            if model_type == 'hrm':
                # HRM returns ponder_loss separately, but it's already included in 'loss'
                # We can log it separately if needed
                ponder_loss = model_output.get('ponder_loss')
                if ponder_loss is not None and hasattr(self, 'log'):
                    self.log('train/ponder_loss', ponder_loss.item() if torch.is_tensor(ponder_loss) else ponder_loss, 
                            on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        
        elif isinstance(model_output, tuple):
            # Models that return tuples (e.g., DeepSeek, GPT, LLaDA)
            if len(model_output) >= 2:
                logits, loss = model_output[0], model_output[1]
                if len(model_output) > 2:
                    router_loss = model_output[2]
            elif len(model_output) == 1:
                logits = model_output[0]
            else:
                 raise ValueError("Model returned an empty tuple.")
        
        elif torch.is_tensor(model_output):
            # Models that return only logits tensor
            logits = model_output
        
        else:
            raise TypeError(f"Unsupported model output type: {type(model_output)}")
            
        return {'logits': logits, 'loss': loss, 'router_loss': router_loss}


    def training_step(self, batch, batch_idx):
        t0 = time.time()
        input_ids, targets = self._unpack_batch(batch)
        loss = None
        router_loss = None
        logits = None

        # --- Simplified Forward Pass ---
        try:
            with self.timing_stats.track("forward"):
                # Run the unified forward pass under autocast
                # Keep tensors on their original device to avoid unnecessary transfers
                input_ids_detached = input_ids
                targets_detached = targets

                forward_kwargs = {}
                if self.args.model_type.lower() == 'llada' and getattr(self.args, 'use_bd3_training', False):
                    forward_kwargs['use_bd3_training'] = True
                use_amp = (self.device.type == 'cuda')
                with torch.amp.autocast(enabled=use_amp, device_type='cuda'):
                    outputs = self(input_ids_detached, targets=targets_detached, **forward_kwargs)
                
                loss = outputs['loss']
                router_loss = outputs.get('router_loss') # Use .get for safety
                # Log auxiliary CE loss if provided by model
                aux_ce = outputs.get('aux_ce_loss')
                if aux_ce is not None and torch.is_tensor(aux_ce):
                    self.log('train/aux_ce_loss', aux_ce.item(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
                # Log components if provided
                if 'score_entropy_loss' in outputs and outputs['score_entropy_loss'] is not None:
                    se = outputs['score_entropy_loss']
                    self.log('train/se_loss', se.item() if torch.is_tensor(se) else se, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
                if 'masked_ce_loss' in outputs and outputs['masked_ce_loss'] is not None:
                    mce = outputs['masked_ce_loss']
                    self.log('train/masked_ce_loss', mce.item() if torch.is_tensor(mce) else mce, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

            # If loss is not calculated by the model, compute it now
            if loss is None:
                logits = outputs['logits']
                if logits is None:
                     raise ValueError("Model output did not contain 'loss' or 'logits'.")
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets_detached.view(-1), ignore_index=-100)

            # --- Loss Combination and NaN/Inf Handling ---
            combined_loss = loss
            if router_loss is not None and torch.is_tensor(router_loss):
                if not (torch.isnan(router_loss).any() or torch.isinf(router_loss).any()):
                    # For models that don't internally add router_loss, add it here
                    if self.args.model_type.lower() not in ['mla', 'parscale_mla', 'moe_mla']:
                        router_loss_coef = getattr(self.args, 'router_z_loss_coef', 0.001)
                        combined_loss = combined_loss + router_loss_coef * router_loss
                    self.log('train/router_loss', router_loss.item(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
                else:
                    print(f"WARNING: NaN/Inf detected in router_loss at step {self.global_step}. Setting to zero.")
                    router_loss = torch.zeros_like(router_loss)


            if torch.isnan(combined_loss).any() or torch.isinf(combined_loss).any():
                lr = self.trainer.optimizers[0].param_groups[0]['lr']
                grad_norm_val = self.trainer.callback_metrics.get('train/grad_norm', 'N/A')
                print(f"ERROR: NaN/Inf detected in combined_loss at step {self.global_step}.")
                print(f"Details: LR={lr:.2e}, Last Grad Norm={grad_norm_val}, Raw Loss={loss.item() if loss is not None else 'N/A'}")
                self.log('train/nan_loss_skipped', 1.0, on_step=True, on_epoch=False, sync_dist=True)
                # Return a zero loss with gradient to prevent crash
                combined_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        except Exception as e:
            if _CheckpointStop is not None and isinstance(e, _CheckpointStop):
                # Let torch.utils.checkpoint handle its internal control flow
                raise
            print(f"Error during training step {self.global_step}: {e}")
            print(traceback.format_exc())
            cleanup_memory()
            self.log('train/step_error', 1.0, on_step=True, on_epoch=False, sync_dist=True)
            # Create a dummy loss to prevent crashing
            combined_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss = torch.tensor(0.0, device=self.device) # for logging

        # --- Logging ---
        dt = time.time() - t0
        loss_value = loss.item()
        self.log('train/loss', loss_value, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        
        # Wandb logging
        if self.use_wandb and self.global_rank == 0:
            wandb_metrics = {
                'train/loss': loss_value,
                'train/combined_loss': combined_loss.item(),
                'train/step_time_ms': dt * 1000,
                'learning_rate': self.trainer.optimizers[0].param_groups[0]['lr'],
                'global_step': self.global_step
            }
        
        # Calculate gradient norms for monitoring
        grad_norm = 0.0
        # Schedule this to run after the backward pass via a hook if possible,
        # but for simplicity, checking after the optimizer step is also fine.
        # This is just a snapshot of the *previous* step's gradients.
        if self.global_step % 10 == 0:
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** 0.5
            self.log('train/grad_norm', grad_norm, on_step=True, on_epoch=False, sync_dist=True)

        self.log('train/combined_loss', combined_loss.item(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log('train/step_time_ms', dt * 1000, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

        # Token/s calculation (now uses non-padding tokens for accuracy)
        non_pad_tokens_mask = (targets != -100)
        batch_tokens = non_pad_tokens_mask.sum().item() # Total non-pad tokens in the batch

        # Percentage of non-padding tokens in the batch
        total_tokens_in_batch = targets.numel()
        non_padding_token_count = batch_tokens
        non_padding_token_percentage = (
            (non_padding_token_count / total_tokens_in_batch) * 100.0 if total_tokens_in_batch > 0 else 0.0
        )

        self.total_tokens += batch_tokens * self.trainer.world_size
        self.tokens_window.append((time.time(), batch_tokens * self.trainer.world_size))
        if len(self.tokens_window) > self.window_size:
            self.tokens_window.pop(0)

        if len(self.tokens_window) > 1:
            window_time = self.tokens_window[-1][0] - self.tokens_window[0][0]
            window_tokens = sum(tokens for _, tokens in self.tokens_window)
            current_tokens_per_sec = window_tokens / window_time if window_time > 0 else 0
        else:
            current_tokens_per_sec = 0

        self.log('tokens_per_sec_step', current_tokens_per_sec, on_step=True, on_epoch=False, prog_bar=True, sync_dist=False)
        self.log('train/non_pad_pct', non_padding_token_percentage, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log('total_tokens', float(self.total_tokens), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        # Calculate average sequence length from the same mask
        avg_seq_len = non_pad_tokens_mask.sum(dim=1).float().mean().item()
        self.log('train/avg_seq_len', avg_seq_len, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        # CSV Logging
        self._buffer_metrics_for_csv(loss_value, grad_norm, current_tokens_per_sec, avg_seq_len)
        
        # Complete wandb logging with additional metrics
        if self.use_wandb and self.global_rank == 0:
            wandb_metrics.update({
                'train/grad_norm': grad_norm,
                'tokens_per_sec': current_tokens_per_sec,
                'train/non_pad_pct': non_padding_token_percentage,
                'total_tokens': float(self.total_tokens),
                'train/avg_seq_len': avg_seq_len,
            })
            
            # Add router loss if available
            if router_loss is not None and torch.is_tensor(router_loss):
                wandb_metrics['train/router_loss'] = router_loss.item()
            
            try:
                wandb.log(wandb_metrics, step=self.global_step)
            except Exception as e:
                print(f"Warning: Failed to log to wandb: {e}")

        # Memory diagnostic after first batch (to detect early imbalances)
        if self.global_step == 1 and self.trainer.world_size > 1:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                max_allocated = torch.cuda.max_memory_allocated() / 1024**3
                print(f"\n[Rank {self.global_rank}] VRAM after first training step:")
                print(f"  - Allocated: {allocated:.2f}GB")
                print(f"  - Reserved: {reserved:.2f}GB")
                print(f"  - Max allocated: {max_allocated:.2f}GB")

                # Check for large tensors on this device
                import gc
                large_tensors = []
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj):
                            if obj.device.type == 'cuda' and obj.device.index == self.global_rank:
                                size_gb = obj.element_size() * obj.nelement() / 1024**3
                                if size_gb > 0.5:  # Only show tensors > 0.5GB
                                    large_tensors.append((size_gb, obj.shape, obj.dtype))
                    except:
                        pass

                if large_tensors:
                    large_tensors.sort(reverse=True)
                    print(f"  - Large tensors (>0.5GB) on GPU {self.global_rank}:")
                    for size_gb, shape, dtype in large_tensors[:5]:  # Top 5
                        print(f"    {size_gb:.2f}GB: shape={shape}, dtype={dtype}")

        # Periodic tasks
        if self.global_step > 0 and self.global_step % 200 == 0:
            if self.global_rank == 0:
                self.generate_sample_text()
                self._log_metrics_to_csv()
            
            if self.args.model_type == 'parscale_mla' and hasattr(self.model, 'analyze_stream_diversity'):
                diversity_stats = self.model.analyze_stream_diversity()
                self.log_dict({f'parscale/{k}': v for k,v in diversity_stats.items()}, on_step=True, on_epoch=False, sync_dist=True)

        # Clear caches less frequently to retain useful caches
        if self.global_step % 50 == 0:
            if hasattr(self.model, 'clear_cache'):
                self.model.clear_cache()
            elif hasattr(self.model, 'cache_manager'):
                self.model.cache_manager.clear()
        
        return combined_loss


    def validation_step(self, batch, batch_idx):
        input_ids, targets = self._unpack_batch(batch)

        with torch.no_grad():
            self.model.eval()
            use_amp = (self.device.type == 'cuda')
            with torch.amp.autocast(enabled=use_amp, device_type='cuda'):
                outputs = self(input_ids, targets=targets)
            loss = outputs['loss']
            router_loss = outputs.get('router_loss')
            aux_ce = outputs.get('aux_ce_loss')
            if aux_ce is not None and torch.is_tensor(aux_ce):
                self.log('val/aux_ce_loss', aux_ce.item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            if 'score_entropy_loss' in outputs and outputs['score_entropy_loss'] is not None:
                se = outputs['score_entropy_loss']
                self.log('val/se_loss', se.item() if torch.is_tensor(se) else se, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            if 'masked_ce_loss' in outputs and outputs['masked_ce_loss'] is not None:
                mce = outputs['masked_ce_loss']
                self.log('val/masked_ce_loss', mce.item() if torch.is_tensor(mce) else mce, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

            if loss is None:
                logits = outputs['logits']
                if logits is None:
                    # Log error and use a high loss value as a fallback
                    print(f"VAL_STEP [{self.current_epoch}/{batch_idx}]: Model output missing 'loss' and 'logits'.")
                    loss = torch.tensor(10.0, device=self.device)
                else:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        
        # Log validation metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        perplexity = calculate_perplexity(loss)
        self.log('val/perplexity', perplexity, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        if router_loss is not None:
             self.log('val/router_loss', router_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # Wandb validation logging
        if self.use_wandb and self.global_rank == 0:
            val_metrics = {
                'val/loss': loss.item() if torch.is_tensor(loss) else loss,
                'val/perplexity': perplexity,
                'epoch': self.current_epoch
            }
            
            if router_loss is not None:
                val_metrics['val/router_loss'] = router_loss.item() if torch.is_tensor(router_loss) else router_loss
            
            # Add auxiliary losses if available
            if aux_ce is not None and torch.is_tensor(aux_ce):
                val_metrics['val/aux_ce_loss'] = aux_ce.item()
            
            try:
                wandb.log(val_metrics, step=self.global_step)
            except Exception as e:
                print(f"Warning: Failed to log validation metrics to wandb: {e}")

        return loss

    def configure_optimizers(self):
        """Sets up the optimizer and learning rate scheduler."""
        # Force foreach=False in DDP to prevent memory imbalance
        # foreach=True can cause different memory allocation patterns per rank
        force_foreach_false = self.trainer.world_size > 1 if hasattr(self, 'trainer') else False

        # Check if we should use FP8-optimized optimizer
        use_fp8_optimizer = (
            getattr(self.args, 'use_fp8', False) and
            getattr(self.args, 'optimizer_type', 'adamw') == 'adamw'
        )

        if use_fp8_optimizer:
            # Use FP8AdamW with low-precision moments
            print("Using FP8AdamW optimizer with BF16 moments")
            
            # Separate parameters by precision requirements
            high_precision_params = []
            standard_params = []
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if any(keep in name.lower() for keep in ['embed', 'norm', 'head', 'rope']):
                        high_precision_params.append(param)
                    else:
                        standard_params.append(param)
            
            # Create parameter groups
            param_groups = []
            
            if high_precision_params:
                param_groups.append({
                    'params': high_precision_params,
                    'lr': self.args.learning_rate,
                    'name': 'high_precision'
                })
            
            if standard_params:
                param_groups.append({
                    'params': standard_params,
                    'lr': self.args.learning_rate,
                    'name': 'standard'
                })
            
            # Create FP8AdamW optimizer
            optimizer = FP8AdamW(
                param_groups,
                lr=self.args.learning_rate,
                betas=(self.args.beta1, self.args.beta2),
                weight_decay=self.args.weight_decay,
                use_low_precision_moments=True
            )
        elif hasattr(self.model, 'configure_optimizers'):
            # Prepare kwargs for optimizer configuration
            optimizer_kwargs = {}
            
            # Add GaLore-specific parameters if using GaLore
            if hasattr(self.args, 'optimizer_type') and self.args.optimizer_type in ['galore', 'galore-8bit', 'galore2']:
                optimizer_kwargs.update({
                    'galore_rank': getattr(self.args, 'galore_rank', 128),
                    'galore_update_proj_gap': getattr(self.args, 'galore_update_proj_gap', 200),
                    'galore_scale': getattr(self.args, 'galore_scale', 0.25),
                    'galore_proj_type': getattr(self.args, 'galore_proj_type', 'std')
                })
                # Add GaLore2-specific parameters
                if self.args.optimizer_type == 'galore2':
                    optimizer_kwargs['galore_quantize_proj'] = getattr(self.args, 'galore_quantize_proj', None)
            
            # For models with configure_optimizers method, use it
            # Pass force_foreach_false to prevent DDP memory imbalance
            optimizer_kwargs['force_foreach_false'] = force_foreach_false

            optimizer = self.model.configure_optimizers(
                weight_decay=self.args.weight_decay,
                learning_rate=self.args.learning_rate,
                betas=(self.args.beta1, self.args.beta2),
                device_type=self.device.type,
                optimizer_type=getattr(self.args, 'optimizer_type', None),  # Pass optimizer type
                **optimizer_kwargs
            )
            print(f"Using optimizer configured by model: {type(optimizer).__name__}")
        else:
            # Fallback to default AdamW for models without configure_optimizers
            print(f"Using default AdamW optimizer (foreach={not force_foreach_false})")
            try:
                optimizer = AdamW(
                    self.model.parameters(),
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay,
                    betas=(self.args.beta1, self.args.beta2),
                    fused=False,
                    foreach=not force_foreach_false  # Disable foreach in DDP
                )
            except TypeError:
                # Older PyTorch versions don't have foreach parameter
                optimizer = AdamW(
                    self.model.parameters(),
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay,
                    betas=(self.args.beta1, self.args.beta2)
                )

        # Learning rate scheduler
        if self.args.decay_lr:
            # Check if optimizer is MultiOptimizer - if so, skip scheduler
            if hasattr(optimizer, '__class__') and optimizer.__class__.__name__ == 'MultiOptimizer':
                print("MultiOptimizer detected - skipping LR scheduler (each sub-optimizer manages its own LR)")
                return optimizer
            else:
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
        pass


    def on_save_checkpoint(self, checkpoint):
        """Persist running token counters for resume."""
        try:
            checkpoint['total_tokens'] = float(self.total_tokens)
            checkpoint['tokens_window_buffer'] = [
                (float(ts), float(tok)) for ts, tok in self.tokens_window
            ]
        except Exception as exc:
            print(f"Warning: failed to record token counters in checkpoint: {exc}")


    def on_load_checkpoint(self, checkpoint):
        """Restore token counters when resuming from checkpoint."""
        total_tokens = checkpoint.get('total_tokens')
        if total_tokens is not None:
            try:
                self.total_tokens = float(total_tokens)
            except (TypeError, ValueError):
                print(f"Warning: invalid total_tokens value in checkpoint: {total_tokens}")

        tokens_window = checkpoint.get('tokens_window_buffer')
        if tokens_window is not None:
            try:
                self.tokens_window = [
                    (float(ts), float(tok)) for ts, tok in tokens_window
                ]
            except Exception as exc:
                print(f"Warning: failed to restore tokens_window from checkpoint: {exc}")


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

            with torch.no_grad():
                 use_amp = (self.device.type == 'cuda')
                 with torch.amp.autocast(enabled=use_amp, device_type='cuda'):
                     output_text = generate_text(
                         self.model,
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

        return input_ids, targets
    
    def _init_csv_logging(self):
        """Initialize CSV logging with a unique filename."""
        # Build filename from model parameters
        model_type = self.args.model_type
        size = self.args.size
        batch_size = self.args.batch_size
        block_size = self.args.block_size
        
        # Add precision info
        precision = "fp32"
        if hasattr(self.args, 'use_fp8') and self.args.use_fp8:
            precision = "fp8"
        elif hasattr(self.args, 'dtype'):
            if 'bfloat16' in str(self.args.dtype):
                precision = "bf16"
            elif 'float16' in str(self.args.dtype):
                precision = "fp16"
        
        # Add compile info
        compile_str = "compile" if hasattr(self.args, 'compile') and self.args.compile else "no-compile"
        
        # Add optimizer info
        optimizer_str = getattr(self.args, 'optimizer_type', 'adamw')
        
        # Add dataset info if available
        dataset_str = getattr(self.args, 'dataset', 'apollo-mini')
        
        # Create base filename
        base_filename = f"{model_type}_{size}_{batch_size}_{block_size}_{precision}_{compile_str}_{optimizer_str}_{dataset_str}"
        
        # Ensure we always log under outputs/metrics_logs regardless of model output dir
        default_metrics_dir = os.path.join('outputs', 'metrics_logs')
        log_dir = getattr(self.args, 'metrics_log_dir', default_metrics_dir)
        log_dir = os.path.abspath(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        legacy_log_dir = None
        output_root = getattr(self.args, 'output_dir', None) or getattr(self.args, 'out_dir', None)
        if output_root:
            legacy_log_dir = os.path.abspath(os.path.join(output_root, 'metrics_logs'))

        resume_mode = getattr(self.args, 'init_from', 'scratch') == 'resume'
        existing_file = None

        if resume_mode:
            search_dirs = [log_dir]
            if legacy_log_dir and os.path.isdir(legacy_log_dir) and legacy_log_dir not in search_dirs:
                search_dirs.append(legacy_log_dir)

            for candidate_dir in search_dirs:
                if not os.path.isdir(candidate_dir):
                    continue

                candidates = sorted(
                    f for f in os.listdir(candidate_dir)
                    if f.startswith(base_filename) and f.endswith('.csv')
                )

                if not candidates:
                    continue

                candidate_path = os.path.join(candidate_dir, candidates[-1])

                if candidate_dir != log_dir:
                    target_path = os.path.join(log_dir, os.path.basename(candidate_path))
                    try:
                        shutil.copy2(candidate_path, target_path)
                        print(
                            "Copied metrics log from legacy location to outputs/metrics_logs:"
                            f" {candidate_path} -> {target_path}"
                        )
                        existing_file = target_path
                    except IOError as e:
                        print(f"Failed to copy legacy metrics log {candidate_path}: {e}")
                        existing_file = None
                else:
                    existing_file = candidate_path

                if existing_file:
                    break

        if existing_file:
            self.csv_file_path = existing_file
            try:
                self.csv_file = open(self.csv_file_path, 'a', newline='')
                self.csv_writer = csv.writer(self.csv_file)
                print(f"CSV metrics logging resumed (appending): {self.csv_file_path}")
            except IOError as e:
                print(f"Error opening existing CSV for append: {e}")
                self.csv_file = None
                self.csv_writer = None
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"{base_filename}_{timestamp}.csv"
            self.csv_file_path = os.path.join(log_dir, csv_filename)

            counter = 1
            while os.path.exists(self.csv_file_path):
                csv_filename = f"{base_filename}_{timestamp}_{counter}.csv"
                self.csv_file_path = os.path.join(log_dir, csv_filename)
                counter += 1

            try:
                self.csv_file = open(self.csv_file_path, 'w', newline='')
                self.csv_writer = csv.writer(self.csv_file)
                self.csv_writer.writerow(['step', 'train_loss', 'val_loss', 'val_perplexity', 'learning_rate', 'tokens_per_sec', 'avg_seq_len', 'batch_size', 'block_size', 'total_tokens', 'grad_norm', 'timestamp'])
                self.csv_file.flush()
                print(f"CSV metrics logging initialized: {self.csv_file_path}")
            except IOError as e:
                print(f"Error initializing CSV logging: {e}")
                self.csv_file = None
                self.csv_writer = None

    def _buffer_metrics_for_csv(self, loss, grad_norm, tps, avg_seq_len):
        """Helper to buffer metrics for CSV logging."""
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        current_val_loss = self.trainer.callback_metrics.get('val/loss')
        current_val_perplexity = self.trainer.callback_metrics.get('val/perplexity')

        if current_val_loss is not None:
            self.last_val_loss = current_val_loss
        if current_val_perplexity is not None:
            self.last_val_perplexity = current_val_perplexity
        
        self.metrics_buffer.append([
            self.global_step,
            f"{loss:.6f}",
            f"{self.last_val_loss:.6f}" if self.last_val_loss is not None else "N/A",
            f"{self.last_val_perplexity:.6f}" if self.last_val_perplexity is not None else "N/A",
            f"{lr:.2e}",
            f"{tps:.2f}",
            f"{avg_seq_len:.2f}",
            f"{self.total_tokens}",
            f"{grad_norm:.4f}" if grad_norm > 0 else "N/A",
            timestamp
        ])

    def _log_metrics_to_csv(self):
        """Write buffered metrics to CSV file."""
        if not self.csv_writer or not self.metrics_buffer:
            return
        
        try:
            num_rows = len(self.metrics_buffer)
            self.csv_writer.writerows(self.metrics_buffer)
            self.csv_file.flush()
            self.metrics_buffer = []
            if self.global_rank == 0:
                print(f"Written {num_rows} rows to CSV (up to step {self.global_step})")
        except IOError as e:
            print(f"Error writing to CSV file: {e}")


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

                 # Setup progressive training if enabled
                 if self.progressive_training:
                     print(f"Progressive training enabled:")
                     print(f"  Block sizes: {self.progressive_block_sizes}")
                     print(f"  Epochs per stage: {self.progressive_epochs_per_stage}")
                     print(f"  LR scale at transitions: {self.progressive_lr_scale}")
                     self.setup_progressive_training()

             # Multi-GPU optimizations
             if self.trainer.world_size > 1:
                 if self.global_rank == 0:
                     print(f"\n=== Multi-GPU Setup ===")
                     print(f"Number of GPUs: {self.trainer.world_size}")
                     print(f"Strategy: {self.trainer.strategy.__class__.__name__}")
                     print(f"Per-GPU batch size: {self.args.batch_size}")
                     print(f"Effective global batch size: {self.args.batch_size * self.trainer.world_size * self.trainer.accumulate_grad_batches}")

                 # Ensure model is properly distributed
                 if hasattr(self.model, 'train'):
                     self.model.train()

                 # Synchronize all processes
                 if torch.cuda.is_available():
                     torch.cuda.synchronize()

                 # Force empty cache to free reserved memory after DDP setup
                 # This helps balance VRAM reserved between ranks
                 if torch.cuda.is_available():
                     torch.cuda.set_device(self.global_rank)
                     torch.cuda.empty_cache()
                     torch.cuda.synchronize()
                     if self.global_rank == 0:
                         print("  - Cleared CUDA cache to balance reserved memory")

                 # Diagnostic: Print detailed VRAM usage per rank to detect imbalances
                 if torch.cuda.is_available():
                     allocated = torch.cuda.memory_allocated() / 1024**3
                     reserved = torch.cuda.memory_reserved() / 1024**3
                     max_allocated = torch.cuda.max_memory_allocated() / 1024**3
                     print(f"[Rank {self.global_rank}] VRAM after setup:")
                     print(f"  - Allocated: {allocated:.2f}GB")
                     print(f"  - Reserved: {reserved:.2f}GB")
                     print(f"  - Max allocated: {max_allocated:.2f}GB")

                     # Count model parameters on this device
                     model_params = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**3
                     print(f"  - Model params size: {model_params:.2f}GB")

                 if self.global_rank == 0:
                     print(f"\n=== Multi-GPU setup completed ===\n")

                     # Give warning if VRAM imbalance detected (will be checked after first batch)
                     print(f"Note: VRAM usage will be monitored for imbalances.")
                     print(f"      If GPU 1 consistently has >5GB more than GPU 0, there's an issue.\n")


    def teardown(self, stage=None):
        if stage == 'fit' or stage is None:
            print("Tearing down the trainer...")
             
            # Properly close dataloaders
            if hasattr(self.trainer, 'train_dataloader') and hasattr(self.trainer.train_dataloader.dataset, 'close'):
                print("Closing training dataset...")
                self.trainer.train_dataloader.dataset.close()

            if hasattr(self.trainer, 'val_dataloaders') and self.trainer.val_dataloaders:
                for dl in self.trainer.val_dataloaders:
                    if hasattr(dl, 'dataset') and hasattr(dl.dataset, 'close'):
                        print("Closing validation dataset...")
                        dl.dataset.close()
            
            # Final CSV log flush
            if self.global_rank == 0:
                self._log_metrics_to_csv()

            # Code to run after training finishes
            cleanup_memory()
            
            # Close CSV file
            if self.csv_file:
                self.csv_file.close()
                print(f"CSV logging closed: {self.csv_file_path}")
            
            if self.global_rank == 0:
                print("Training finished. Final memory stats:")
                print_memory_stats("Teardown")
                
                # Finish wandb run
                if self.use_wandb:
                    try:
                        wandb.finish()
                        print("Wandb run finished successfully")
                    except Exception as e:
                        print(f"Warning: Failed to finish wandb run: {e}")


# Note: DataModule definition would go here or in a separate file.
# For now, we assume dataloaders are passed directly to trainer.fit()
