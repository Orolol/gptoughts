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
from models.deepseek.deepseek_adapter_mtp import DeepSeekMiniMTP, DeepSeekMiniConfigMTP
from models.llada.model import LLaDAModel
from models.models.model import GPT
from models.models.mla_model import MLAModel, MLAModelConfig
from models.models.parscale_mla import ParScaleMLA, ParScaleMLAConfig, create_parscale_mla
from models.models.mla_selective_model import MLASelectiveModel, MLASelectiveModelConfig
from models.models.moe_mla_model import MOEMLA, MOEMLAConfig
from models.models.nsa_model import NSAModel, NSAModelConfig
from models.models.hse_model import HSEModel, HSEConfig
from models.models.hrm_model import HRM, HRMConfig
from models.mdm.model import MDMModel
from models.config import MDMConfig
from models.sedd.model import SEDDModel, SEDDConfig
from train.train_utils import (
    get_lr, calculate_perplexity, ensure_model_dtype,
    AveragedTimingStats, generate_text, estimate_loss
)
from optimization.memory_optim import cleanup_memory, print_memory_stats, preallocate_cuda_memory
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
            config = self._create_deepseek_config()
            model = DeepSeekMiniMTP(config)
        elif model_type == 'llada':
            config = self._create_llada_config()
            model = LLaDAModel(config)
        elif model_type == 'sedd':
            config = self._create_sedd_config()
            model = SEDDModel(config)
        elif model_type == 'mla':
            config = self._create_mla_config()
            model = self._create_mla_model(config)
        elif model_type == 'mla_selective':
            config = self._create_mla_selective_config()
            model = self._create_mla_selective_model(config)
        elif model_type == 'parscale_mla':
            config = self._create_parscale_mla_config()
            model = self._create_parscale_mla_model(config)
        elif model_type == 'mla_llada':
            config = self._create_mla_llada_config()
            model = self._create_mla_llada_model(config)
        elif model_type == 'mdm':
            config = self._create_mdm_config()
            model = self._create_mdm_model(config)
        elif model_type == 'moe_mla':
            config = self._create_moe_mla_config()
            model = self._create_moe_mla_model(config)
        elif model_type == 'slm':
            config = self._create_slm_config()
            model = self._create_slm_model(config)
        elif model_type == 'nsa':
            config = self._create_nsa_config()
            model = self._create_nsa_model(config)
        elif model_type == 'hse':
            config = self._create_hse_config()
            model = self._create_hse_model(config)
        elif model_type == 'swan':
            config = self._create_swan_config()
            from models.models.swan_model import SWANModel
            model = SWANModel(config)
        elif model_type == 'swa_mla':
            config = self._create_swa_mla_config()
            from models.models.swa_mla_model import SWAMLAModel
            model = SWAMLAModel(config)
        elif model_type == 'hrm':
            config = self._create_hrm_config()
            model = self._create_hrm_model(config)
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
        
        # Common BD3 parameters
        bd3_params = {}
        if hasattr(self.args, 'bd3_block_length'):
            bd3_params['bd3_block_length'] = self.args.bd3_block_length
        if hasattr(self.args, 'bd3_beta'):
            bd3_params['bd3_beta'] = self.args.bd3_beta
        if hasattr(self.args, 'bd3_omega'):
            bd3_params['bd3_omega'] = self.args.bd3_omega
        if hasattr(self.args, 'disable_entropy_regularization'):
            bd3_params['disable_entropy_regularization'] = self.args.disable_entropy_regularization
        
        if self.args.size == 'small':
            config = LLaDAConfig(
                block_size=self.args.block_size, vocab_size=self.args.vocab_size,
                n_layer=8, n_head=8, n_embd=768, dropout=self.args.dropout,
                bias=self.args.bias, ratio_kv=8, use_checkpoint=False,
                **bd3_params
            )
        elif self.args.size == 'medium':
            config = LLaDAConfig(
                block_size=self.args.block_size, vocab_size=self.args.vocab_size,
                n_layer=16, n_head=16, n_embd=1024, dropout=self.args.dropout,
                bias=self.args.bias, ratio_kv=8, use_checkpoint=False,
                **bd3_params
            )
        else: # large
            config = LLaDAConfig(
                block_size=self.args.block_size, vocab_size=self.args.vocab_size,
                n_layer=24, n_head=16, n_embd=1536, dropout=self.args.dropout,
                bias=self.args.bias, ratio_kv=8, use_checkpoint=False,
                **bd3_params
            )
        return config

    def _create_sedd_config(self):
        if self.args.size == 'small':
            config = SEDDConfig(
                block_size=self.args.block_size,
                vocab_size=self.args.vocab_size,
                n_layer=12,
                n_head=12,
                n_embd=768,
                dropout=self.args.dropout,
                bias=self.args.bias,
                mask_token_id=self.args.vocab_size - 1,
                cond_dim=128,
                scale_by_sigma=True,
                mlp_ratio=4,
                graph_type=getattr(self.args, 'graph_type', 'absorb'),
                noise_type=getattr(self.args, 'noise_type', 'loglinear'),
                sigma_min=getattr(self.args, 'sigma_min', 1e-4),
                sigma_max=getattr(self.args, 'sigma_max', 20.0),
                use_gradient_checkpointing=False,
                attention_backend=getattr(self.args, 'attention_backend', None),
                use_fp8=getattr(self.args, 'use_fp8', False),
                use_dyt=getattr(self.args, 'use_dyt', False),
                dyt_alpha_init=getattr(self.args, 'dyt_alpha_init', 0.5),
            )
        elif self.args.size == 'medium':
            config = SEDDConfig(
                block_size=self.args.block_size,
                vocab_size=self.args.vocab_size,
                n_layer=16,
                n_head=16,
                n_embd=1024,
                dropout=self.args.dropout,
                bias=self.args.bias,
                mask_token_id=self.args.vocab_size - 1,
                cond_dim=128,
                scale_by_sigma=True,
                mlp_ratio=4,
                graph_type=getattr(self.args, 'graph_type', 'absorb'),
                noise_type=getattr(self.args, 'noise_type', 'loglinear'),
                sigma_min=getattr(self.args, 'sigma_min', 1e-4),
                sigma_max=getattr(self.args, 'sigma_max', 20.0),
                use_gradient_checkpointing=False,
                attention_backend=getattr(self.args, 'attention_backend', None),
                use_fp8=getattr(self.args, 'use_fp8', False),
                use_dyt=getattr(self.args, 'use_dyt', False),
                dyt_alpha_init=getattr(self.args, 'dyt_alpha_init', 0.5),
            )
        else:  # large
            config = SEDDConfig(
                block_size=self.args.block_size,
                vocab_size=self.args.vocab_size,
                n_layer=24,
                n_head=16,
                n_embd=1536,
                dropout=self.args.dropout,
                bias=self.args.bias,
                mask_token_id=self.args.vocab_size - 1,
                cond_dim=128,
                scale_by_sigma=True,
                mlp_ratio=4,
                graph_type=getattr(self.args, 'graph_type', 'absorb'),
                noise_type=getattr(self.args, 'noise_type', 'loglinear'),
                sigma_min=getattr(self.args, 'sigma_min', 1e-4),
                sigma_max=getattr(self.args, 'sigma_max', 20.0),
                use_gradient_checkpointing=False,
                attention_backend=getattr(self.args, 'attention_backend', None),
                use_fp8=getattr(self.args, 'use_fp8', False),
                use_dyt=getattr(self.args, 'use_dyt', False),
                dyt_alpha_init=getattr(self.args, 'dyt_alpha_init', 0.5),
            )
        return config

    def _create_mla_config(self):
        """Create configuration for MLA-Model."""
        # Define key parameters based on size
        if self.args.size == 'small':
            n_layer = 12
            n_embd = 768
            n_head = 16
            q_lora_rank = 0
            kv_lora_rank = 256
            qk_nope_head_dim = 128
            qk_rope_head_dim = 64
            v_head_dim = 128
        elif self.args.size == 'medium':
            n_layer = 16
            n_embd = 1024
            n_head = 16
            q_lora_rank = 0
            kv_lora_rank = 512
            qk_nope_head_dim = 128
            qk_rope_head_dim = 64
            v_head_dim = 128
        elif self.args.size == 'large':
            n_layer = 32
            n_embd = 2048
            n_head = 32
            q_lora_rank = 0
            kv_lora_rank = 512
            qk_nope_head_dim = 192
            qk_rope_head_dim = 96
            v_head_dim = 192
        else:  # xl
            n_layer = 40
            n_embd = 2560
            n_head = 20
            q_lora_rank = 0
            kv_lora_rank = 1024
            qk_nope_head_dim = 384
            qk_rope_head_dim = 192
            v_head_dim = 384
            
            
        
        # Create config object
        config = MLAModelConfig(
            # Architecture
            n_layer=n_layer,
            n_embd=n_embd,
            n_head=n_head,
            vocab_size=self.args.vocab_size,
            block_size=self.args.block_size,
            
            # MLA parameters
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            
            # MoE parameters
            use_moe=False,  # Set to False for dense model
            
            # RoPE parameters
            rope_theta=10000.0,
            
            # Precision
            fp8_params=getattr(self.args, 'use_fp8', False),
            fp8_mla_params=getattr(self.args, 'fp8_mla_params', False),
            use_fp8=getattr(self.args, 'use_fp8', False),
            fp8_tile_size=getattr(self.args, 'fp8_tile_size', 128),
            
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
    
    def _create_mla_selective_config(self):
        """Create configuration for MLA-Selective Model."""
        # First get base MLA config
        base_config = self._create_mla_config()
        
        # Create MLA Selective config with base MLA parameters
        config = MLASelectiveModelConfig(
            # Copy base MLA parameters
            n_layer=base_config.n_layer,
            n_embd=base_config.n_embd,
            n_head=base_config.n_head,
            vocab_size=base_config.vocab_size,
            block_size=base_config.block_size,
            q_lora_rank=base_config.q_lora_rank,
            kv_lora_rank=base_config.kv_lora_rank,
            qk_nope_head_dim=base_config.qk_nope_head_dim,
            qk_rope_head_dim=base_config.qk_rope_head_dim,
            v_head_dim=base_config.v_head_dim,
            rope_theta=base_config.rope_theta,
            fp8_params=base_config.fp8_params,
            fp8_mla_params=base_config.fp8_mla_params,
            dropout=base_config.dropout,
            bias=base_config.bias,
            attention_backend=base_config.attention_backend,
            # Disable gradient checkpointing if using torch.compile
            use_gradient_checkpointing=base_config.use_gradient_checkpointing,

        )
        
        return config
    
    def _create_mla_selective_model(self, config):
        """Create MLA-Selective Model instance with the given configuration."""
        return MLASelectiveModel(config)
    
    def _create_parscale_mla_config(self):
        """Create configuration for ParScale-MLA model."""
        # First get base MLA config
        base_config = self._create_mla_config()
        
        # Create ParScale config with MLA parameters
        config = ParScaleMLAConfig(
            # Copy base MLA parameters
            n_layer=base_config.n_layer,
            n_embd=base_config.n_embd,
            n_head=base_config.n_head,
            vocab_size=base_config.vocab_size,
            block_size=base_config.block_size,
            q_lora_rank=base_config.q_lora_rank,
            kv_lora_rank=base_config.kv_lora_rank,
            qk_nope_head_dim=base_config.qk_nope_head_dim,
            qk_rope_head_dim=base_config.qk_rope_head_dim,
            v_head_dim=base_config.v_head_dim,
            use_moe=base_config.use_moe,
            rope_theta=base_config.rope_theta,
            fp8_params=base_config.fp8_params,
            fp8_mla_params=base_config.fp8_mla_params,
            dropout=base_config.dropout,
            bias=base_config.bias,
            attention_backend=base_config.attention_backend,
            use_gradient_checkpointing=base_config.use_gradient_checkpointing,
            
            # Add ParScale specific parameters
            parallel_streams=getattr(self.args, 'parallel_streams', 8),
            prefix_length=getattr(self.args, 'prefix_length', 48),
            latent_prefix_length=getattr(self.args, 'latent_prefix_length', 16),
            aggregator_epsilon=getattr(self.args, 'aggregator_epsilon', 0.1),
            diversity_weight=getattr(self.args, 'diversity_weight', 0.1),
            use_dynamic_inference=getattr(self.args, 'use_dynamic_inference', True),
            complexity_threshold=getattr(self.args, 'complexity_threshold', 0.5),
            freeze_base_in_stage2=getattr(self.args, 'freeze_base_in_stage2', True),
        )
        
        return config
    
    def _create_parscale_mla_model(self, config):
        """Create ParScale-MLA model instance."""
        # Check if we're in stage 2 and have a base checkpoint
        if getattr(self.args, 'training_stage', 1) == 2 and getattr(self.args, 'base_checkpoint', None):
            print(f"Loading base model from checkpoint: {self.args.base_checkpoint}")
            # Load base model checkpoint
            checkpoint = torch.load(self.args.base_checkpoint, map_location='cpu')
            
            # Extract base model state dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # Remove 'model.' prefix if present
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            else:
                state_dict = checkpoint
            
            # Create base MLA model and load weights
            base_model = MLAModel(config)
            base_model.load_state_dict(state_dict, strict=False)
            
            # Create ParScale model with pre-trained base
            model = ParScaleMLA(base_model=base_model, config=config)
        else:
            # Create ParScale model from scratch
            model = create_parscale_mla(size=self.args.size, parallel_streams=config.parallel_streams)
        
        # Set training stage
        model.set_training_stage(getattr(self.args, 'training_stage', 1))
        
        return model
    
    def _create_mla_llada_config(self):
        """Create configuration for MLA-LLaDA model."""
        from models.models.mla_llada import MLALLaDAConfig
        
        # Base dimensions based on size - optimized for target parameter counts
        # Note: With 128k vocab, embeddings+head take ~2*vocab_size*hidden_size parameters
        if self.args.size == 'small':  # Target: ~500M parameters
            hidden_size = 768  # Increased from 256
            num_layers = 12   # Reduced from 20
            intermediate_size = 1024  # Increased proportionally
            kv_lora_rank = 64  # Increased from 32
        elif self.args.size == 'medium':  # Target: ~1B parameters
            hidden_size = 384
            num_layers = 24
            intermediate_size = 1536
            kv_lora_rank = 48
        elif self.args.size == 'large':  # Target: ~2B parameters
            hidden_size = 512
            num_layers = 32
            intermediate_size = 2048
            kv_lora_rank = 64
        else:  # xl - Target: ~4B parameters
            hidden_size = 768
            num_layers = 40
            intermediate_size = 2560
            kv_lora_rank = 96
            
        config = MLALLaDAConfig(
            # Base GPTConfig parameters
            n_embd=hidden_size,
            n_layer=num_layers,
            vocab_size=self.args.vocab_size,
            block_size=self.args.block_size,
            dropout=self.args.dropout,
            bias=self.args.bias,
            
            # MLA-LLaDA specific
            hidden_size=hidden_size,
            num_layers=num_layers,
            
            # MLA parameters
            q_lora_rank=0,  # Full rank for queries
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            
            # LLaDA parameters  
            mask_token_id=self.args.vocab_size - 1,  # Use last valid token ID
            max_diffusion_steps=50,
            min_diffusion_steps=10,
            mask_ratio_min=getattr(self.args, 'mask_ratio_min', 0.15),
            mask_ratio_max=getattr(self.args, 'mask_ratio_max', 0.85),
            remasking_strategy=getattr(self.args, 'remasking_strategy', 'low_confidence'),
            
            # FP8 configuration
            use_fp8=getattr(self.args, 'use_fp8', False),
            fp8_format='e4m3',
            
            # DynamicTanh
            use_dyt=getattr(self.args, 'use_dyt', False),
            dyt_alpha_init=getattr(self.args, 'dyt_alpha_init', 0.5),
            
            # General parameters
            intermediate_size=intermediate_size,
            gradient_checkpointing=getattr(self.args, 'gradient_checkpointing', True),
        )
        
        return config
    
    def _create_mla_llada_model(self, config):
        """Create MLA-LLaDA model instance."""
        from models.models.mla_llada import create_mla_llada_model
        return create_mla_llada_model(config)

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
    
    def _create_mdm_config(self):
        """Create configuration for Masked Diffusion Model."""
        if self.args.size == 'small':
            config = MDMConfig(
                n_layer=12,
                n_head=12,
                n_embd=768,
                block_size=self.args.block_size,
                vocab_size=self.args.vocab_size,
                dropout=self.args.dropout,
                bias=self.args.bias,
                mask_token_id=self.args.vocab_size - 1,
                attention_backend=getattr(self.args, 'attention_backend', None)
            )
        elif self.args.size == 'medium':
            config = MDMConfig(
                n_layer=24,
                n_head=16,
                n_embd=1024,
                block_size=self.args.block_size,
                vocab_size=self.args.vocab_size,
                dropout=self.args.dropout,
                bias=self.args.bias,
                mask_token_id=self.args.vocab_size - 1,
                attention_backend=getattr(self.args, 'attention_backend', None)
            )
        elif self.args.size == 'large':
            config = MDMConfig(
                n_layer=32,
                n_head=16,
                n_embd=1536,
                block_size=self.args.block_size,
                vocab_size=self.args.vocab_size,
                dropout=self.args.dropout,
                bias=self.args.bias,
                mask_token_id=self.args.vocab_size - 1,
                attention_backend=getattr(self.args, 'attention_backend', None)
            )
        else:  # xl
            config = MDMConfig(
                n_layer=40,
                n_head=20,
                n_embd=2560,
                block_size=self.args.block_size,
                vocab_size=self.args.vocab_size,
                dropout=self.args.dropout,
                bias=self.args.bias,
                mask_token_id=self.args.vocab_size - 1,
                attention_backend=getattr(self.args, 'attention_backend', None)
            )
        return config
    
    def _create_mdm_model(self, config):
        """Create MDM model instance."""
        return MDMModel(config)
    
    def _create_moe_mla_config(self):
        """Create configuration for MOE-MLA model."""
        # Start with base MLA config dimensions
        base_config = self._create_mla_config()
        
        # Create MOE-MLA config
        config = MOEMLAConfig(
            # Copy base MLA parameters
            n_layer=base_config.n_layer,
            n_embd=base_config.n_embd,
            n_head=base_config.n_head,
            vocab_size=base_config.vocab_size,
            block_size=base_config.block_size,
            q_lora_rank=base_config.q_lora_rank,
            kv_lora_rank=base_config.kv_lora_rank,
            qk_nope_head_dim=base_config.qk_nope_head_dim,
            qk_rope_head_dim=base_config.qk_rope_head_dim,
            v_head_dim=base_config.v_head_dim,
            rope_theta=base_config.rope_theta,
            dropout=base_config.dropout,
            bias=base_config.bias,
            attention_backend=base_config.attention_backend,
            use_gradient_checkpointing=base_config.use_gradient_checkpointing,
            
            # MOE-specific parameters
            num_experts=getattr(self.args, 'num_experts', 16),
            experts_per_token=getattr(self.args, 'experts_per_token', 2),
            shared_weight_ratio=getattr(self.args, 'shared_weight_ratio', 0.9),
            
            # FP8 settings
            use_fp8=getattr(self.args, 'use_fp8', False),
            fp8_tile_size=getattr(self.args, 'fp8_tile_size', 128),
            
            # DyT settings
            use_dyt=getattr(self.args, 'use_dyt', False),
            dyt_alpha_init=getattr(self.args, 'dyt_alpha_init', 0.5),
        )
        
        return config
    
    def _create_moe_mla_model(self, config):
        """Create MOE-MLA model instance."""
        return MOEMLA(config)
    
    def _create_slm_config(self):
        """Create configuration for SLM-MoE-MLA Model."""
        from models.models.slm_moe_mla import SLMConfig
        
        # Define key parameters based on size - optimized for different parameter counts
        if self.args.size == 'tiny':
            # ~100M parameters
            n_layer = 8
            n_embd = 512
            n_head = 8
            num_experts = 16
            experts_per_token = 2
            kv_lora_rank = 128
            qk_nope_head_dim = 64
            qk_rope_head_dim = 32
            v_head_dim = 64
        elif self.args.size == 'small':
            # ~200M parameters
            n_layer = 8
            n_embd = 768
            n_head = 12
            num_experts = 32
            experts_per_token = 1
            kv_lora_rank = 256
            qk_nope_head_dim = 96
            qk_rope_head_dim = 32
            v_head_dim = 96
        elif self.args.size == 'medium':
            # ~400M parameters
            n_layer = 12
            n_embd = 1024
            n_head = 16
            num_experts = 48
            experts_per_token = 6
            kv_lora_rank = 384
            qk_nope_head_dim = 128
            qk_rope_head_dim = 64
            v_head_dim = 128
        else:  # large - ~800M parameters
            n_layer = 16
            n_embd = 1280
            n_head = 20
            num_experts = 64
            experts_per_token = 8
            kv_lora_rank = 512
            qk_nope_head_dim = 160
            qk_rope_head_dim = 80
            v_head_dim = 160
        
        # Create SLM config
        config = SLMConfig(
            # Architecture
            n_layer=n_layer,
            n_embd=n_embd,
            n_head=n_head,
            vocab_size=self.args.vocab_size,
            block_size=self.args.block_size,
            
            # MLA parameters
            q_lora_rank=0,  # No low-rank for queries in SLM
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            
            # MoE parameters - ultra high sharing
            num_experts=getattr(self.args, 'num_experts', num_experts),
            experts_per_token=getattr(self.args, 'experts_per_token', experts_per_token),
            shared_weight_ratio=getattr(self.args, 'shared_weight_ratio', 0.90),
            
            # Router parameters
            router_temperature=getattr(self.args, 'router_temperature', 0.1),
            router_z_loss_coef=getattr(self.args, 'router_z_loss_coef', 0.001),
            load_balance_coef=getattr(self.args, 'load_balance_coef', 0.01),
            
            # RoPE parameters
            rope_theta=10000.0,
            original_max_seq_len=self.args.block_size,
            
            # Precision and optimization
            use_fp8=getattr(self.args, 'use_fp8', False),
            fp8_tile_size=getattr(self.args, 'fp8_tile_size', 128),
            
            # Training parameters
            dropout=self.args.dropout,
            bias=self.args.bias,
            use_gradient_checkpointing=getattr(self.args, 'gradient_checkpointing', True),
            
            # Dynamic Tanh - enabled by default for SLM
            use_dyt=getattr(self.args, 'use_dyt', True),
            dyt_alpha_init=getattr(self.args, 'dyt_alpha_init', 0.5),
            
            # Attention backend
            attention_backend=getattr(self.args, 'attention_backend', None),
        )
        
        return config
    
    def _create_slm_model(self, config):
        """Create SLM-MoE-MLA model instance."""
        from models.models.slm_moe_mla import SLMMLA
        return SLMMLA(config)
    
    def _create_nsa_config(self):
        """Create configuration for NSA Model."""
        # Define key parameters based on size
        if self.args.size == 'small':
            n_layer = 12
            n_embd = 768
            n_head = 12
            compress_block_size = 16
            compress_stride = 8
            selection_block_size = 32
            num_selected_blocks = 8
            sliding_window_size = 256
        elif self.args.size == 'medium':
            n_layer = 24
            n_embd = 1024
            n_head = 16
            compress_block_size = 32
            compress_stride = 16
            selection_block_size = 64
            num_selected_blocks = 12
            sliding_window_size = 384
        elif self.args.size == 'large':
            n_layer = 32
            n_embd = 2048
            n_head = 16
            compress_block_size = 32
            compress_stride = 16
            selection_block_size = 64
            num_selected_blocks = 16
            sliding_window_size = 512
        else:  # xl
            n_layer = 40
            n_embd = 2560
            n_head = 20
            compress_block_size = 64
            compress_stride = 32
            selection_block_size = 128
            num_selected_blocks = 20
            sliding_window_size = 640
        
        # Create config object
        config = NSAModelConfig(
            # Architecture
            n_layer=n_layer,
            n_embd=n_embd,
            n_head=n_head,
            vocab_size=self.args.vocab_size,
            block_size=self.args.block_size,
            
            # NSA specific parameters
            compress_block_size=compress_block_size,
            compress_stride=compress_stride,
            selection_block_size=selection_block_size,
            num_selected_blocks=num_selected_blocks,
            sliding_window_size=sliding_window_size,
            
            # Common training parameters
            dropout=self.args.dropout,
            bias=self.args.bias,
            use_gradient_checkpointing=getattr(self.args, 'gradient_checkpointing', True),
            use_fp8=self.args.use_fp8,
            fp8_tile_size=getattr(self.args, 'fp8_tile_size', 128),
            
            # DyT options
            use_dyt=getattr(self.args, 'use_dyt', False),
            dyt_alpha_init=getattr(self.args, 'dyt_alpha_init', 0.5),
            
            # Label smoothing
            label_smoothing=getattr(self.args, 'label_smoothing', 0.0),
        )
        
        return config
    
    def _create_nsa_model(self, config):
        """Create NSA model instance."""
        return NSAModel(config)

    def _create_hse_config(self):
        """Create configuration for HSE Model (NSA orchestrator + MoE experts)."""
        # Size presets similar to NSA
        if self.args.size == 'small':
            n_layer, n_embd, n_head = 12, 768, 12
            compress_block_size, compress_stride = 16, 8
            selection_block_size, num_selected_blocks = 32, 8
            sliding_window_size = 256
        elif self.args.size == 'medium':
            n_layer, n_embd, n_head = 24, 1024, 16
            compress_block_size, compress_stride = 32, 16
            selection_block_size, num_selected_blocks = 64, 12
            sliding_window_size = 384
        elif self.args.size == 'large':
            n_layer, n_embd, n_head = 32, 2048, 16
            compress_block_size, compress_stride = 32, 16
            selection_block_size, num_selected_blocks = 64, 16
            sliding_window_size = 512
        else:  # xl
            n_layer, n_embd, n_head = 40, 2560, 20
            compress_block_size, compress_stride = 64, 32
            selection_block_size, num_selected_blocks = 128, 20
            sliding_window_size = 640

        config = HSEConfig(
            # Architecture
            n_layer=n_layer,
            n_embd=n_embd,
            n_head=n_head,
            vocab_size=self.args.vocab_size,
            block_size=self.args.block_size,

            # Orchestrator (NSA)
            compress_block_size=compress_block_size,
            compress_stride=compress_stride,
            selection_block_size=selection_block_size,
            num_selected_blocks=num_selected_blocks,
            sliding_window_size=sliding_window_size,

            # Experts
            num_experts=getattr(self.args, 'num_experts', 8),
            experts_per_token=getattr(self.args, 'experts_per_token', 2),

            # Scribes
            scribe_chunk_size=getattr(self.args, 'scribe_chunk_size', 2048),
            scribe_summary_len=getattr(self.args, 'scribe_summary_len', 128),

            # QAP budgets
            qap_per_step=getattr(self.args, 'qap_per_step', 12),
            qap_per_expert=getattr(self.args, 'qap_per_expert', 6),
            qap_max_queries=getattr(self.args, 'qap_max_queries', 20),

            # Training
            dropout=self.args.dropout,
            bias=self.args.bias,
            use_gradient_checkpointing=getattr(self.args, 'gradient_checkpointing', True),
            use_fp8=getattr(self.args, 'use_fp8', False),
            fp8_tile_size=getattr(self.args, 'fp8_tile_size', 128),
            use_dyt=getattr(self.args, 'use_dyt', False),
            dyt_alpha_init=getattr(self.args, 'dyt_alpha_init', 0.5),
            label_smoothing=getattr(self.args, 'label_smoothing', 0.0),
            # Standard attention extras
            ratio_kv=getattr(self.args, 'ratio_kv', 8),
            attention_backend=getattr(self.args, 'attention_backend', None),
        )
        return config

    def _create_hse_model(self, config):
        """Create HSE model instance."""
        return HSEModel(config)

    def _create_swan_config(self):
        from models.models.swan_model import SWANConfig

        size_defaults = {
            'small': dict(n_layer=16, n_embd=1024, n_head=16),
            'medium': dict(n_layer=24, n_embd=1536, n_head=16),
            'large': dict(n_layer=28, n_embd=2048, n_head=24),
            'xl': dict(n_layer=32, n_embd=4096, n_head=32),
        }
        size_key = getattr(self.args, 'size', 'medium')
        size_config = size_defaults.get(size_key, size_defaults['medium'])

        n_layer = getattr(self.args, 'n_layer', None)
        if n_layer is None:
            n_layer = size_config['n_layer']
        n_head = getattr(self.args, 'n_head', None)
        if n_head is None:
            n_head = size_config['n_head']
        n_embd = getattr(self.args, 'n_embd', None)
        if n_embd is None:
            n_embd = size_config['n_embd']

        global_layers = getattr(self.args, 'global_layers_per_cycle', None)
        if global_layers is None:
            global_layers = 1
        local_layers = getattr(self.args, 'local_layers_per_cycle', None)
        if local_layers is None:
            local_layers = 3
        swa_window = getattr(self.args, 'swa_window', None)
        if swa_window is None:
            swa_window = 512

        logit_scale_base = getattr(self.args, 'logit_scale_base', None)
        if logit_scale_base is None:
            logit_scale_base = 128.0
        logit_scale_window = getattr(self.args, 'logit_scale_window', None)
        if logit_scale_window is None:
            logit_scale_window = 128
        logit_scale_offset = getattr(self.args, 'logit_scale_offset', None)
        if logit_scale_offset is None:
            logit_scale_offset = 0
        logit_scale_min = getattr(self.args, 'logit_scale_min', None)
        if logit_scale_min is None:
            logit_scale_min = 1.0
        logit_scale_max = getattr(self.args, 'logit_scale_max', None)

        config = SWANConfig(
            vocab_size=self.args.vocab_size,
            block_size=self.args.block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=self.args.dropout,
            bias=self.args.bias,
            ratio_kv=getattr(self.args, 'ratio_kv', 1),
            attention_backend=getattr(self.args, 'attention_backend', None),
            use_gradient_checkpointing=getattr(self.args, 'use_gradient_checkpointing', getattr(self.args, 'gradient_checkpointing', True)),
            global_layers_per_cycle=global_layers,
            local_layers_per_cycle=local_layers,
            swa_window=swa_window,
            rope_theta=getattr(self.args, 'rope_theta', 10000.0),
            logit_scale_base=logit_scale_base,
            logit_scale_window=logit_scale_window,
            logit_scale_offset=logit_scale_offset,
            logit_scale_min=logit_scale_min,
            logit_scale_max=logit_scale_max,
            apply_logit_scale_during_training=getattr(self.args, 'apply_logit_scale_during_training', False),
            label_smoothing=getattr(self.args, 'label_smoothing', 0.0),
        )

        return config

    def _create_swa_mla_config(self):
        from models.models.swa_mla_model import SWAMLAConfig

        size_defaults = {
            'small': dict(n_layer=12, n_embd=1024, n_head=16),
            'medium': dict(n_layer=24, n_embd=1536, n_head=16),
            'large': dict(n_layer=28, n_embd=2048, n_head=24),
            'xl': dict(n_layer=32, n_embd=4096, n_head=32),
        }
        size_key = getattr(self.args, 'size', 'medium')
        size_config = size_defaults.get(size_key, size_defaults['medium'])

        n_layer = getattr(self.args, 'n_layer', None) or size_config['n_layer']
        n_head = getattr(self.args, 'n_head', None) or size_config['n_head']
        n_embd = getattr(self.args, 'n_embd', None) or size_config['n_embd']

        swa_layers = getattr(self.args, 'swa_layers_per_cycle', None)
        if swa_layers is None:
            swa_layers = getattr(self.args, 'local_layers_per_cycle', 2)
        mla_layers = getattr(self.args, 'mla_layers_per_cycle', None)
        if mla_layers is None:
            mla_layers = getattr(self.args, 'global_layers_per_cycle', 1)

        logit_scale_base = getattr(self.args, 'logit_scale_base', None)
        logit_scale_window = getattr(self.args, 'logit_scale_window', None)
        if logit_scale_window is None:
            logit_scale_window = 128
        logit_scale_offset = getattr(self.args, 'logit_scale_offset', None)
        if logit_scale_offset is None:
            logit_scale_offset = 0
        logit_scale_min = getattr(self.args, 'logit_scale_min', None)
        if logit_scale_min is None:
            logit_scale_min = 1.0
        logit_scale_max = getattr(self.args, 'logit_scale_max', None)

        config = SWAMLAConfig(
            vocab_size=self.args.vocab_size,
            block_size=self.args.block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=self.args.dropout,
            bias=self.args.bias,
            ratio_kv=getattr(self.args, 'ratio_kv', 1),
            attention_backend=getattr(self.args, 'attention_backend', None),
            use_gradient_checkpointing=getattr(self.args, 'use_gradient_checkpointing', getattr(self.args, 'gradient_checkpointing', True)),
            use_dyt=getattr(self.args, 'use_dyt', False),
            dyt_alpha_init=getattr(self.args, 'dyt_alpha_init', 0.5),
            swa_layers_per_cycle=swa_layers,
            mla_layers_per_cycle=mla_layers,
            swa_window=getattr(self.args, 'swa_window', 256),
            rope_theta=getattr(self.args, 'rope_theta', 10000.0),
            logit_scale_base=logit_scale_base,
            logit_scale_window=logit_scale_window,
            logit_scale_offset=logit_scale_offset,
            logit_scale_min=logit_scale_min,
            logit_scale_max=logit_scale_max,
            apply_logit_scale_during_training=getattr(self.args, 'apply_logit_scale_during_training', False),
            q_lora_rank=getattr(self.args, 'mla_q_lora_rank', 0),
            kv_lora_rank=getattr(self.args, 'mla_kv_lora_rank', 512),
            qk_nope_head_dim=getattr(self.args, 'mla_qk_nope_head_dim', 128),
            qk_rope_head_dim=getattr(self.args, 'mla_qk_rope_head_dim', 64),
            v_head_dim=getattr(self.args, 'mla_v_head_dim', 128),
            attn_impl=getattr(self.args, 'mla_attn_impl', 'absorb'),
            world_size=getattr(self.args, 'devices', 1) if isinstance(getattr(self.args, 'devices', -1), int) else 1,
            rope_scaling=getattr(self.args, 'mla_rope_scaling', None),
            rope_factor=getattr(self.args, 'mla_rope_factor', 1.0),
            mscale=getattr(self.args, 'mla_mscale', 1.0),
            use_fp8=getattr(self.args, 'use_fp8', False),
            fp8_mla_params=getattr(self.args, 'fp8_mla_params', False),
            fp8_tile_size=getattr(self.args, 'fp8_tile_size', 128),
            label_smoothing=getattr(self.args, 'label_smoothing', 0.0),
        )

        rope_scaling = getattr(self.args, 'mla_rope_scaling', None)
        if rope_scaling is not None and isinstance(rope_scaling, dict):
            config.rope_scaling = rope_scaling

        return config
    
    # --- End Config Creation Methods ---

    def _parse_bool_arg(self, arg_name, default=False):
        """Parse boolean argument that might be passed as string."""
        if not hasattr(self.args, arg_name):
            return default
        value = getattr(self.args, arg_name)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value)

    def _create_hrm_config(self):
        """Create configuration for HRM (Hierarchical Reasoning Model)."""
        # Define sizes matching the paper's ~27M params for small
        if self.args.size == 'small':
            n_embd = 512
            n_head = 8
            n_inner = 2048
            max_segments = 8
            cycles_per_segment = 2
            steps_per_cycle = 3
        elif self.args.size == 'medium':
            n_embd = 768
            n_head = 12
            n_inner = 3072
            max_segments = 8
            cycles_per_segment = 3
            steps_per_cycle = 4
        elif self.args.size == 'large':
            n_embd = 1024
            n_head = 16
            n_inner = 4096
            max_segments = 10
            cycles_per_segment = 4
            steps_per_cycle = 4
        else:  # xl
            n_embd = 1536
            n_head = 24
            n_inner = 6144
            max_segments = 12
            cycles_per_segment = 4
            steps_per_cycle = 5

        config = HRMConfig(
            # Architecture
            n_layer=1,  # HRM uses recurrence, not stacked layers
            n_embd=n_embd,
            n_head=n_head,
            n_inner=n_inner,
            vocab_size=self.args.vocab_size,
            block_size=self.args.block_size,
            
            # HRM specific parameters
            cycles_per_segment=getattr(self.args, 'hrm_cycles_per_segment', cycles_per_segment) if hasattr(self.args, 'hrm_cycles_per_segment') and self.args.hrm_cycles_per_segment is not None else cycles_per_segment,
            steps_per_cycle=getattr(self.args, 'hrm_steps_per_cycle', steps_per_cycle) if hasattr(self.args, 'hrm_steps_per_cycle') and self.args.hrm_steps_per_cycle is not None else steps_per_cycle,
            max_segments=getattr(self.args, 'hrm_max_segments', max_segments) if hasattr(self.args, 'hrm_max_segments') and self.args.hrm_max_segments is not None else max_segments,
            min_segments=getattr(self.args, 'min_segments', 1),
            gradient_steps=getattr(self.args, 'hrm_gradient_steps', -1) if hasattr(self.args, 'hrm_gradient_steps') and self.args.hrm_gradient_steps is not None else -1,
            
            # ACT parameters
            use_act=self._parse_bool_arg('hrm_use_act', True),
            act_epsilon=getattr(self.args, 'act_epsilon', 0.1),
            ponder_loss_weight=getattr(self.args, 'ponder_loss_weight', 0.01),
            halt_bias_init=getattr(self.args, 'halt_bias_init', -2.0),
            
            # Training parameters
            dropout=self.args.dropout,
            bias=self.args.bias,
            use_gradient_checkpointing=False,  # HRM uses 1-step gradient instead
            deq_one_step=getattr(self.args, 'hrm_deq_one_step', False),
            use_deep_supervision=getattr(self.args, 'hrm_use_deep_supervision', False),
            n_supervision_segments=getattr(self.args, 'hrm_n_supervision_segments', 4),
            label_smoothing=getattr(self.args, 'label_smoothing', 0.0),
            
            # Learning rate and optimization
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            warmup_steps=self.args.warmup_iters,
            grad_clip=getattr(self.args, 'grad_clip', 1.0),
        )
        return config

    def _create_hrm_model(self, config):
        """Create HRM model instance."""
        return HRM(config)
    
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
        elif model_type in ['mla', 'mla_selective', 'parscale_mla', 'moe_mla', 'slm', 'nsa']:
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

        # Periodic tasks
        if self.global_step > 0 and self.global_step % 1000 == 0:
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
            print(f"Using default AdamW optimizer")
            try:
                optimizer = AdamW(
                    self.model.parameters(),
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay,
                    betas=(self.args.beta1, self.args.beta2),
                    fused=False
                )
            except TypeError:
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
