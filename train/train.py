"""
Script d'entraînement générique pour les modèles LLM.
Ce script peut être utilisé pour entraîner différents types de modèles (DeepSeek, LLaDA, etc.)
Optimisé pour maximiser l'utilisation du GPU.
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
import csv
import shutil
from datetime import datetime
import wandb


import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group, barrier as dist_barrier
from torch.amp import GradScaler

# Essayer d'importer les fonctions d'optimisation GPU avancées
try:
    from optimization.cuda_optim import (
        setup_cuda_optimizations,
        print_gpu_stats,
        optimize_attention_operations
    )
    from optimization.memory_optim import (
        cleanup_memory, 
        print_memory_stats,
        preallocate_cuda_memory
    )
    ENHANCED_OPTIMIZATIONS = True
except ImportError:
    # Fallback sur les optimisations standard
    from optimization.cuda_optim import (
        setup_cuda_optimizations,
        print_gpu_stats
    )
    from optimization.memory_optim import (
        cleanup_memory, 
        print_memory_stats
    )
    # Ces fonctions peuvent ne pas être disponibles dans la version standard
    preallocate_cuda_memory = lambda: print("preallocate_cuda_memory not available")
    optimize_attention_operations = lambda: print("optimize_attention_operations not available")
    ENHANCED_OPTIMIZATIONS = False

# Import des fonctions utilitaires
from train.train_utils import (
    get_gpu_count, setup_distributed, reduce_metrics, calculate_perplexity,
    get_lr, cleanup_old_checkpoints, ensure_model_dtype,
    save_checkpoint, load_checkpoint, find_latest_checkpoint,
    get_context_manager, AveragedTimingStats, generate_text
)

# Import model configs
from train.model_configs import *

# Import models
from models.sedd.model import SEDDModel

# Import FP8 optimizer
try:
    from optimization.fp8_deepseek_trainer import FP8AdamW, FP8MixedPrecisionTrainer
    FP8_AVAILABLE = True
except ImportError:
    FP8_AVAILABLE = False
    print("FP8 training not available (fp8_deepseek_trainer not found)")

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

        # Progressive training parameters
        self.progressive_training = getattr(args, 'progressive_training', False)
        self.progressive_block_sizes = getattr(args, 'progressive_block_sizes', [512, 1024, 2048, 4096])
        self.progressive_epochs_per_stage = getattr(args, 'progressive_epochs_per_stage', 5)
        self.progressive_current_stage = 0
        self.progressive_lr_scale = getattr(args, 'progressive_lr_scale', 0.5)

        # Block size adaptation parameters
        self.load_checkpoint_path = getattr(args, 'load_checkpoint_path', None)
        self.original_block_size = getattr(args, 'original_block_size', None)

        # CSV logging attributes
        self.metrics_buffer = []
        self.csv_file_path = None
        self.csv_writer = None
        self.csv_file = None
        self.last_val_loss = None
        self.last_val_perplexity = None

        # Wandb
        self.use_wandb = getattr(args, 'use_wandb', True)

        self.setup_environment()
        self.setup_model()
        self.setup_datasets()
        self.setup_training()
        
    def setup_environment(self):
        """Configure l'environnement d'exécution (DDP, device, etc.) avec optimisations avancées"""
        # Distributed setup
        self.ddp = int(os.environ.get('RANK', -1)) != -1
        if self.ddp:
            self.ddp_rank, self.ddp_local_rank, self.ddp_world_size, self.device = setup_distributed(backend='nccl')
            self.master_process = self.ddp_rank == 0
            self.seed_offset = self.ddp_rank
        else:
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1
            # Allow explicit device override (useful for testing on CPU)
            if hasattr(self.args, 'device') and self.args.device:
                self.device = self.args.device
                print(f"Using explicitly specified device: {self.device}")
            else:
                self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # Create output directory
        if self.master_process:
            os.makedirs(self.args.output_dir, exist_ok=True)
        
        # Set random seed
        torch.manual_seed(1337 + self.seed_offset)
        
        # Setup device and dtype
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        
        # Determine dtype based on model type and available hardware
        if hasattr(self.args, 'dtype') and self.args.dtype is not None:
            self.dtype = self.args.dtype
            print(f"Using dtype: {self.dtype}")
        else:
            # Choisir le meilleur dtype en fonction du GPU
            if torch.cuda.is_available():
                # Check if FP8 is supported
                fp8_supported = False
                try:
                    # First check if transformer_engine is properly installed and can be loaded
                    try:
                        import transformer_engine
                        # Check if we're running on H100 or later GPU that supports FP8
                        if torch.cuda.get_device_properties(0).major >= 9 or (
                            torch.cuda.get_device_properties(0).major == 8 and 
                            torch.cuda.get_device_properties(0).minor >= 6
                        ):
                            fp8_supported = True
                    except (ImportError, ModuleNotFoundError, RuntimeError) as e:
                        print(f"transformer_engine not available or couldn't be loaded: {e}")
                        print("FP8 support will be disabled")
                except Exception as e:
                    print(f"Error checking for FP8 support: {e}")
                    pass
                
                if fp8_supported and hasattr(self.args, 'use_fp8') and self.args.use_fp8:
                    self.dtype = 'fp8'        # H100 et plus récent
                elif torch.cuda.is_bf16_supported():
                    self.dtype = 'bfloat16'  # Meilleur pour les GPUs récents (Ampere+)
                else:
                    self.dtype = 'float16'   # Pour les GPUs plus anciens
            else:
                self.dtype = 'float32'       # CPU
            
        print(f"Using dtype: {self.dtype}")
        
        # Setup pytorch dtype equivalent - FP8 doesn't have a direct PyTorch equivalent
        if self.dtype == 'fp8':
            # For model parameters we'll use BF16, but computation will be in FP8
            self.ptdtype = torch.bfloat16
            print("Using FP8 for computation with BF16 for parameters")
        else:
            self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        
        # Apply CUDA optimizations if available
        if torch.cuda.is_available():
            setup_cuda_optimizations()
            
            # Optimiser les opérations d'attention si les optimisations avancées sont disponibles
            if ENHANCED_OPTIMIZATIONS and hasattr(self.args, 'optimize_attention') and self.args.optimize_attention:
                optimize_attention_operations()
            
            # Préallouer la mémoire CUDA si demandé
            if hasattr(self.args, 'preallocate_memory') and self.args.preallocate_memory:
                preallocate_cuda_memory()
                
            # Afficher les statistiques GPU détaillées
            if self.master_process:
                print_gpu_stats()
        
        # Calculate tokens per iteration for logging
        self.tokens_per_iter = self.args.batch_size * self.args.block_size * self.args.gradient_accumulation_steps * self.ddp_world_size
        print(f"Tokens per iteration: {self.tokens_per_iter:,}")
        print(f"Batch size: {self.args.batch_size}, block size: {self.args.block_size}, gradient accumulation steps: {self.args.gradient_accumulation_steps}")
        
        # Disable deterministic algorithms for better performance and to prevent NaN issues
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        # Configurer les optimisations de mémoire CUDA
        if torch.cuda.is_available():
            # Réserver un pourcentage plus élevé de la mémoire pour PyTorch
            torch.cuda.set_per_process_memory_fraction(0.95)

            # Configurer l'allocateur CUDA pour une meilleure gestion de la mémoire
            if hasattr(torch.cuda, 'memory_stats'):
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"

        # Initialize wandb if enabled
        if self.use_wandb and self.master_process:
            self._init_wandb()

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
            # Check if we should use MTP variant
            use_mtp = getattr(self.args, 'use_mtp', True)
            
            if use_mtp:
                # Use DeepSeek with Multi-Token Prediction support
                from models.deepseek import DeepSeekMiniMTP, DeepSeekMiniConfigMTP
                # Create config based on model size with MTP parameters
                config = self.create_deepseek_mtp_config()
                self.model = DeepSeekMiniMTP(config)
                print(f"Initialized DeepSeek model with Multi-Token Prediction (MTP) support")
            else:
                # Use standard DeepSeek
                from models.deepseek import DeepSeekMiniTrainable, DeepSeekMiniConfig
                # Create config based on model size
                config = self.create_deepseek_config()
                self.model = DeepSeekMiniTrainable(config)
                print(f"Initialized DeepSeek model without MTP")
            
        elif model_type == 'llada':
            from models.llada.model import LLaDAModel, LLaDAConfig
            # Create config based on model size
            config = self.create_llada_config()
            self.model = LLaDAModel(config)
            
        elif model_type == 'mla':
            from models.models.mla_model import MLAModel, MLAModelConfig, create_mla_model
            # Create config based on model size
            config = self.create_mla_config()
            self.model = MLAModel(config)
            
        elif model_type == 'mla_llada':
            from models.models.mla_llada import MLALLaDAModel, MLALLaDAConfig, create_mla_llada_model
            # Create config based on model size
            config = self.create_mla_llada_config()
            self.model = MLALLaDAModel(config)

        elif model_type == 'sedd':
            # SEDD model
            config = create_sedd_config(self.args)
            self.model = SEDDModel(config)

        elif model_type == 'mdm':
            # MDM model
            config = create_mdm_config(self.args)
            self.model = create_mdm_model(config)

        elif model_type == 'moe_mla':
            # MoE MLA model
            config = create_moe_mla_config(self.args)
            self.model = create_moe_mla_model(config)

        elif model_type == 'slm':
            # SLM model
            config = create_slm_config(self.args)
            self.model = create_slm_model(config)

        elif model_type == 'nsa':
            # NSA model
            config = create_nsa_config(self.args)
            self.model = create_nsa_model(config)

        elif model_type == 'hse':
            # HSE model
            config = create_hse_config(self.args)
            self.model = create_hse_model(config)

        elif model_type == 'swan':
            # SWAN model
            config = create_swan_config(self.args)
            from models.models.swan_model import SWANModel
            self.model = SWANModel(config)

        elif model_type == 'swa_mla':
            # SWA_MLA model
            config = create_swa_mla_config(self.args)
            from models.models.swa_mla_model import SWAMLAModel
            self.model = SWAMLAModel(config)

        elif model_type == 'swa_mla_moe':
            # SWA_MLA_MoE model
            config = create_swa_mla_moe_config(self.args)
            from models.models.swa_mla_moe_model import SWAMLAMOEModel
            self.model = SWAMLAMOEModel(config)

        elif model_type == 'hrm':
            # HRM model
            config = create_hrm_config(self.args)
            self.model = create_hrm_model(config)

        else:
            from models.models.model import GPT, GPTConfig
            # Create config based on model size
            config = self.create_gpt_config()
            self.model = GPT(config)
        
        # Store config for checkpointing
        self.config = config

        # Critical multi-GPU optimization: Force model to CPU first
        # This prevents VRAM imbalance across GPUs in DDP
        if self.ddp and torch.cuda.is_available():
            print("Forcing model to CPU before DDP to prevent VRAM imbalance...")
            self.model = self.model.cpu()
            cleanup_memory()

        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer with specified type if provided
        optimizer_args = {
            'weight_decay': self.args.weight_decay,
            'learning_rate': self.args.learning_rate,
            'betas': (self.args.beta1, self.args.beta2),
            'device_type': self.device_type
        }
        
        # Ajouter le type d'optimiseur s'il est spécifié
        if hasattr(self.args, 'optimizer_type') and self.args.optimizer_type is not None:
            optimizer_args['optimizer_type'] = self.args.optimizer_type
            print(f"Using specified optimizer: {self.args.optimizer_type}")
        
        self.optimizer = self.model.configure_optimizers(**optimizer_args)
        
        # Initialize gradient scaler for mixed precision
        # Note: BFloat16 has sufficient dynamic range and doesn't need gradient scaling
        if self.dtype == 'fp8':
            try:
                import transformer_engine.pytorch as te
                # Pour FP8, désactiver GradScaler complètement
                # transformer_engine gère déjà l'échelle en interne avec fp8_autocast
                self.scaler = GradScaler(enabled=False)
                self.use_unscale = False
                print("Using transformer-engine FP8 integration without GradScaler")
            except (ImportError, ModuleNotFoundError, RuntimeError, Exception) as e:
                # Fallback to BFloat16 which has better stability
                self.scaler = GradScaler(enabled=False)
                self.use_unscale = False
                print(f"Transformer-engine not available or couldn't be loaded: {e}")
                print("Falling back to BFloat16 precision")
                self.dtype = 'bfloat16'
                self.ptdtype = torch.bfloat16
        elif self.dtype == 'float16':
            self.scaler = GradScaler(enabled=True)
            print("Using Float16 with GradScaler")
        else:
            # BF16 and FP32 don't need scaling
            self.scaler = GradScaler(enabled=False)
            if self.dtype == 'bfloat16':
                print("Using BFloat16 without GradScaler (sufficient dynamic range)")
            else:
                print("Using FP32 without GradScaler")
        
        # Initialize training state
        self.iter_num = 1
        self.best_val_loss = float('inf')
        
        # Print model size
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")
        
    def create_deepseek_config(self):
        """Crée une configuration pour le modèle DeepSeek Mini"""
        from models.deepseek import DeepSeekMiniConfig
        
        # Determine model size parameters
        if self.args.size == 'small':
            config = DeepSeekMiniConfig(
                vocab_size=self.args.vocab_size,
                hidden_size=1024,
                num_hidden_layers=8,
                num_attention_heads=8,
                head_dim=128,  # Corrigé pour assurer que num_attention_heads * head_dim == hidden_size
                intermediate_size=2816,
                num_experts=4,  # Réduit de 8 à 4 pour diminuer la consommation de mémoire
                num_experts_per_token=1,  # Réduit de 2 à 1 pour diminuer la consommation de mémoire
                max_position_embeddings=max(16, self.args.block_size),  # Assure un minimum de 16 pour la stabilité
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
    
    def create_deepseek_mtp_config(self):
        """Crée une configuration pour le modèle DeepSeek Mini avec MTP"""
        from models.deepseek import DeepSeekMiniConfigMTP
        
        # Get MTP related arguments or use defaults
        num_mtp_modules = getattr(self.args, 'num_mtp_modules', 1)
        layers_per_mtp = getattr(self.args, 'layers_per_mtp', 1)
        mtp_loss_factor = getattr(self.args, 'mtp_loss_factor', 0.1)
        
        # Determine model size parameters
        if self.args.size == 'small':
            config = DeepSeekMiniConfigMTP(
                vocab_size=self.args.vocab_size,
                hidden_size=1024,
                num_hidden_layers=8,
                num_attention_heads=8,
                head_dim=128,
                intermediate_size=2816,
                num_experts=4,
                num_experts_per_token=1,
                max_position_embeddings=max(16, self.args.block_size),
                kv_compression_dim=64,
                query_compression_dim=192,
                rope_head_dim=32,
                dropout=self.args.dropout,
                attention_dropout=self.args.dropout,
                hidden_dropout=self.args.dropout,
                bias=self.args.bias,
                # MTP specific parameters
                num_mtp_modules=num_mtp_modules,
                layers_per_mtp=layers_per_mtp,
                mtp_loss_factor=mtp_loss_factor,
                use_mtp=True
            )
        elif self.args.size == 'medium':
            config = DeepSeekMiniConfigMTP(
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
                bias=self.args.bias,
                # MTP specific parameters
                num_mtp_modules=num_mtp_modules,
                layers_per_mtp=layers_per_mtp,
                mtp_loss_factor=mtp_loss_factor,
                use_mtp=True
            )
        else:  # large
            config = DeepSeekMiniConfigMTP(
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
                bias=self.args.bias,
                # MTP specific parameters
                num_mtp_modules=num_mtp_modules,
                layers_per_mtp=layers_per_mtp,
                mtp_loss_factor=mtp_loss_factor,
                use_mtp=True
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
                n_layer=16,
                n_head=16,
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
    
    def create_mla_config(self):
        """Crée une configuration pour le modèle MLA"""
        from models.models.mla_model import MLAModelConfig
        
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
    
    def create_mla_llada_config(self):
        """Crée une configuration pour le modèle MLA-LLaDA"""
        from models.models.mla_llada import MLALLaDAConfig
        
        # Define key parameters based on size
        if self.args.size == 'small':
            n_layer = 12
            n_embd = 768
            intermediate_size = 2048
        elif self.args.size == 'medium':
            n_layer = 24
            n_embd = 1024
            intermediate_size = 4096
        elif self.args.size == 'large':
            n_layer = 32
            n_embd = 2048
            intermediate_size = 8192
        else:  # xl
            n_layer = 40
            n_embd = 2560
            intermediate_size = 10240
        
        # Create config object
        config = MLALLaDAConfig(
            # Model dimensions
            hidden_size=n_embd,
            num_layers=n_layer,
            vocab_size=self.args.vocab_size,
            block_size=self.args.block_size,
            
            # MLA specific
            q_lora_rank=0,  # Full rank
            kv_lora_rank=64,  # Compressed latent dimension
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            
            # LLaDA specific
            mask_token_id=self.args.vocab_size - 1,  # Use last token as mask
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
            dyt_init_alpha=getattr(self.args, 'dyt_alpha_init', 0.5),
            
            # Training
            intermediate_size=intermediate_size,
            dropout=self.args.dropout,
            gradient_checkpointing=True,
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
        elif model_type in ['mla', 'mla_selective', 'parscale_mla', 'moe_mla', 'slm', 'nsa', 'swan', 'swa_mla', 'swa_mla_moe']:
            self._adapt_rope_position_encoding(model, old_block_size, new_block_size)
        elif model_type == 'llada':
            self._adapt_llada_position_encoding(model, old_block_size, new_block_size)
        elif model_type in ['deepseek']:
            self._adapt_deepseek_position_encoding(model, old_block_size, new_block_size)
        elif model_type == 'sedd':
            self._adapt_sedd_position_encoding(model, old_block_size, new_block_size)
        elif model_type == 'hrm':
            self._adapt_rope_position_encoding(model, old_block_size, new_block_size)
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

        # Check if we've completed enough iterations for current stage
        # Estimate iterations per stage based on epochs_per_stage
        iters_per_stage = self.progressive_epochs_per_stage * 1000  # Rough estimate

        iters_in_stage = self.iter_num - (self.progressive_current_stage * iters_per_stage)

        return (iters_in_stage >= iters_per_stage and
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
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            param_group['lr'] = old_lr * self.progressive_lr_scale
            print(f"Scaled learning rate: {old_lr:.2e} -> {param_group['lr']:.2e}")

        print(f"=== Stage transition completed ===\n")

        # Log to wandb if available
        if self.use_wandb and self.master_process:
            try:
                wandb.log({
                    'progressive_training/stage': self.progressive_current_stage,
                    'progressive_training/block_size': new_block_size,
                    'progressive_training/transition_iter': self.iter_num
                }, step=self.iter_num)
            except Exception as e:
                print(f"Warning: Failed to log progressive training to wandb: {e}")

        # Save checkpoint after transition
        if self.master_process:
            checkpoint_path = f"progressive_stage_{self.progressive_current_stage}_iter_{self.iter_num}.ckpt"
            self.save_training_checkpoint()
            print(f"Saved checkpoint after stage transition")

        return True

    def _init_csv_logging(self):
        """Initialize CSV logging with a unique filename."""
        if not self.master_process:
            return

        # Build filename from model parameters
        model_type = self.args.model_type
        size = self.args.size
        batch_size = self.args.batch_size
        block_size = self.args.block_size

        # Add precision info
        precision = "fp32"
        if hasattr(self.args, 'use_fp8') and self.args.use_fp8:
            precision = "fp8"
        elif hasattr(self, 'dtype'):
            if 'bfloat16' in str(self.dtype):
                precision = "bf16"
            elif 'float16' in str(self.dtype):
                precision = "fp16"

        # Add compile info
        compile_str = "compile" if hasattr(self.args, 'compile') and self.args.compile else "no-compile"

        # Add optimizer info
        optimizer_str = getattr(self.args, 'optimizer_type', 'adamw')

        # Add dataset info if available
        dataset_str = getattr(self.args, 'dataset', 'apollo-mini')

        # Create base filename
        base_filename = f"{model_type}_{size}_{batch_size}_{block_size}_{precision}_{compile_str}_{optimizer_str}_{dataset_str}"

        # Ensure we always log under outputs/metrics_logs
        default_metrics_dir = os.path.join('outputs', 'metrics_logs')
        log_dir = getattr(self.args, 'metrics_log_dir', default_metrics_dir)
        log_dir = os.path.abspath(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        # Check for existing CSV to resume
        resume_mode = getattr(self.args, 'init_from', 'scratch') == 'resume'
        existing_file = None

        if resume_mode:
            candidates = sorted(
                f for f in os.listdir(log_dir)
                if f.startswith(base_filename) and f.endswith('.csv')
            )
            if candidates:
                existing_file = os.path.join(log_dir, candidates[-1])

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
                self.csv_writer.writerow([
                    'step', 'train_loss', 'val_loss', 'val_perplexity',
                    'learning_rate', 'tokens_per_sec', 'avg_seq_len',
                    'total_tokens', 'grad_norm', 'timestamp'
                ])
                self.csv_file.flush()
                print(f"CSV metrics logging initialized: {self.csv_file_path}")
            except IOError as e:
                print(f"Error initializing CSV logging: {e}")
                self.csv_file = None
                self.csv_writer = None

    def _buffer_metrics_for_csv(self, loss, grad_norm, tps, avg_seq_len, lr):
        """Helper to buffer metrics for CSV logging."""
        if not self.master_process or not self.csv_writer:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.metrics_buffer.append([
            self.iter_num,
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
        if not self.master_process or not self.csv_writer or not self.metrics_buffer:
            return

        try:
            num_rows = len(self.metrics_buffer)
            self.csv_writer.writerows(self.metrics_buffer)
            self.csv_file.flush()
            self.metrics_buffer = []
            print(f"Written {num_rows} rows to CSV (up to step {self.iter_num})")
        except IOError as e:
            print(f"Error writing to CSV file: {e}")

    def load_model_from_checkpoint(self, ckpt_path):
        """Charge un modèle à partir d'un checkpoint"""
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Determine model type
        model_type = self.args.model_type.lower()
        
        if model_type == 'deepseek':
            # Check if we should use MTP variant
            use_mtp = getattr(self.args, 'use_mtp', True)
            
            if use_mtp:
                # Use DeepSeek with Multi-Token Prediction support
                from models.deepseek import DeepSeekMiniMTP, DeepSeekMiniConfigMTP
                
                # Detect if the checkpoint is from an MTP model or not
                if 'num_mtp_modules' in checkpoint.get('model_args', {}):
                    # Create new model with saved MTP config
                    saved_config = DeepSeekMiniConfigMTP(**checkpoint['model_args'])
                else:
                    # Convert non-MTP config to MTP config
                    saved_config = DeepSeekMiniConfigMTP(**checkpoint['model_args'])
                    saved_config.num_mtp_modules = getattr(self.args, 'num_mtp_modules', 1)
                    saved_config.layers_per_mtp = getattr(self.args, 'layers_per_mtp', 1)
                    saved_config.mtp_loss_factor = getattr(self.args, 'mtp_loss_factor', 0.1)
                    saved_config.use_mtp = True
                    print("Converting non-MTP checkpoint to MTP-compatible model")
                
                self.model = DeepSeekMiniMTP(saved_config)
                self.config = saved_config
            else:
                # Use standard DeepSeek
                from models.deepseek import DeepSeekMiniTrainable, DeepSeekMiniConfig
                # Create new model with saved config
                saved_config = DeepSeekMiniConfig(**checkpoint['model_args'])
                self.model = DeepSeekMiniTrainable(saved_config)
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
        
        elif model_type == 'mla':
            from models.models.mla_model import MLAModel, MLAModelConfig
            # Create new model with saved config
            if 'model_args' in checkpoint and isinstance(checkpoint['model_args'], dict):
                saved_config = MLAModelConfig(**checkpoint['model_args'])
            else:
                # Create default config
                saved_config = self.create_mla_config()
                
            self.model = MLAModel(saved_config)
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
            # Initialize optimizer with specified type if provided
            optimizer_args = {
                'weight_decay': self.args.weight_decay,
                'learning_rate': self.args.learning_rate,
                'betas': (self.args.beta1, self.args.beta2),
                'device_type': self.device_type
            }
            
            # Ajouter le type d'optimiseur s'il est spécifié
            if hasattr(self.args, 'optimizer_type') and self.args.optimizer_type is not None:
                optimizer_args['optimizer_type'] = self.args.optimizer_type
                print(f"Using specified optimizer: {self.args.optimizer_type}")
            
            self.optimizer = self.model.configure_optimizers(**optimizer_args)
        
        # Move model to device and ensure correct dtype
        self.model = self.model.to(self.device)
        self.model = ensure_model_dtype(self.model, self.ptdtype)
        
        # Reset scaler
        # Note: BFloat16 has sufficient dynamic range and doesn't need gradient scaling
        self.scaler = GradScaler(enabled=(self.dtype == 'float16'))
        self.scaler.is_enabled = False
        
        cleanup_memory()
    
    def setup_datasets(self):
        """Configure les datasets d'entraînement et de validation"""
        # Import the get_datasets function from data module
        from data.datasets import get_datasets

        # Get datasets - get_datasets() returns dataloaders directly
        if hasattr(self.args, 'tokenizer') and self.args.tokenizer is not None:
            print(f"Tokenizer found: {self.args.tokenizer}")

        self.train_dataset, self.val_dataset = get_datasets(self.args)
        
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

            # Print VRAM diagnostics after DDP initialization
            if torch.cuda.is_available() and self.master_process:
                print("\n=== Multi-GPU VRAM Diagnostics ===")
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
                print("===================================\n")
        
        # Ensure model is in correct precision for stability
        self.model = ensure_model_dtype(self.model, self.ptdtype)
        
        # Use a more conservative gradient clipping value if not specified
        if not hasattr(self.args, 'grad_clip') or self.args.grad_clip == 0.0:
            self.args.grad_clip = 1.0
            print(f"Setting default gradient clipping to {self.args.grad_clip}")
            
        # Enable gradient checkpointing if requested or for large models
        if hasattr(self.args, 'use_gradient_checkpointing') and self.args.use_gradient_checkpointing:
            if hasattr(self.model, 'set_gradient_checkpointing'):
                self.model.set_gradient_checkpointing(True)
                print("Gradient checkpointing enabled")
            elif self.args.size in ['medium', 'large']:
                print("Warning: Gradient checkpointing was requested but model doesn't support it")
        elif self.args.size in ['medium', 'large']:
            # Auto-enable gradient checkpointing for medium and large models
            if hasattr(self.model, 'set_gradient_checkpointing'):
                self.model.set_gradient_checkpointing(True)
                print("Gradient checkpointing automatically enabled for large model")
            
        # Compile model if requested
        if hasattr(self.args, 'compile') and self.args.compile:
            print("Compiling model...")
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
            except Exception as e:
                print(f"Compilation failed: {e}")
                print(traceback.format_exc())
                self.args.compile = False
            print("Compilation finished")
        
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
        print(f"Batch size: {self.args.batch_size}, Block size: {self.args.block_size}")
        print(f"Gradient accumulation steps: {self.args.gradient_accumulation_steps}")
        print_memory_stats("Initial")

        # Initialize CSV logging
        self._init_csv_logging()

        # Setup progressive training if enabled
        if self.progressive_training and self.master_process:
            print(f"Progressive training enabled:")
            print(f"  Block sizes: {self.progressive_block_sizes}")
            print(f"  Epochs per stage: {self.progressive_epochs_per_stage}")
            print(f"  LR scale at transitions: {self.progressive_lr_scale}")
            self.setup_progressive_training()

    def forward(self, input_ids, targets=None, **kwargs):
        """
        Unified forward pass for all model types.
        Returns a standardized dictionary with logits, loss, and optional router_loss.
        """
        model_type = self.args.model_type.lower()

        # Call model forward
        outputs = self.model(input_ids, targets=targets, **kwargs)

        # Standardize output format
        result = {'logits': None, 'loss': None, 'router_loss': None}

        # Handle different output formats from various models
        if isinstance(outputs, dict):
            # Dictionary output (most new models: MLA, MoE, etc.)
            result['logits'] = outputs.get('logits', None)
            result['loss'] = outputs.get('loss', None)
            result['router_loss'] = outputs.get('router_loss', None)

            # Handle HRM ponder loss
            if 'ponder_loss' in outputs:
                result['ponder_loss'] = outputs['ponder_loss']

            # Handle MTP loss for DeepSeek
            if 'mtp_loss' in outputs:
                result['mtp_loss'] = outputs['mtp_loss']

        elif isinstance(outputs, tuple):
            # Tuple output (older models, LLaDA, etc.)
            if len(outputs) == 2:
                result['logits'], result['loss'] = outputs
            elif len(outputs) == 3:
                result['logits'], result['loss'], result['router_loss'] = outputs
            elif len(outputs) == 1:
                result['logits'] = outputs[0]
        else:
            # Single tensor output (just logits, no loss computed)
            result['logits'] = outputs

        return result

    def train(self):
        """Exécute la boucle d'entraînement principale avec optimisations GPU avancées"""
        model_type = self.args.model_type.lower()
        
        # Optimisations pour maximiser l'utilisation du GPU
        if torch.cuda.is_available():
            # Précharger les poids du modèle en mémoire GPU
            for param in self.model.parameters():
                if param.device.type != 'cuda':
                    param.data = param.data.to(self.device)
            
            # Optimiser le scheduler de CUDA
            if hasattr(torch.cuda, 'cudart'):
                torch.cuda.cudart().cudaProfilerStart()
            
            # Synchroniser avant de commencer l'entraînement
            torch.cuda.synchronize()
        
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
            # if self.iter_num % 500 == 0 and self.master_process or self.iter_num == 50:
            # # if self.iter_num 100:
            #     self.generate_sample_text()
            
            # Evaluate model periodically
            if self.iter_num % self.args.eval_interval == 0 and self.master_process and self.iter_num > 0:
                self.evaluate_model()
            
            # Exit if eval_only is set
            if self.iter_num == 0 and self.args.eval_only:
                break
            
            # Forward backward update, with gradient accumulation
            with self.timing_stats.track("optimization"):
                # Utiliser set_to_none=True pour une meilleure performance
                self.optimizer.zero_grad(set_to_none=True)
                total_loss = 0
                total_router_loss = 0
                total_mtp_loss = 0
                skip_optimizer_step = False
                
                try:
                    # Précharger le premier batch en dehors de la boucle pour masquer la latence
                    try:
                        next_batch = next(self.train_iterator)
                    except StopIteration:
                        self.train_iterator = iter(self.train_dataset)
                        next_batch = next(self.train_iterator)
                    
                    for micro_step in range(self.args.gradient_accumulation_steps):
                        if self.ddp:
                            # Synchroniser les gradients seulement à la dernière étape d'accumulation
                            self.model.require_backward_grad_sync = (
                                micro_step == self.args.gradient_accumulation_steps - 1
                            )
                        
                        # Track data loading time
                        with self.timing_stats.track("data_loading"):
                            # Utiliser le batch préchargé
                            batch = next_batch
                            
                            # Précharger le prochain batch en parallèle
                            try:
                                next_batch = next(self.train_iterator)
                            except StopIteration:
                                self.train_iterator = iter(self.train_dataset)
                                next_batch = next(self.train_iterator)
                            
                            # Handle different batch formats
                            if isinstance(batch, dict):
                                # Transférer les données au GPU de manière asynchrone
                                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                                targets = batch.get('labels', input_ids).to(self.device, non_blocking=True)
                            elif isinstance(batch, tuple) and len(batch) >= 2:
                                input_ids = batch[0].to(self.device, non_blocking=True)
                                targets = batch[-1].to(self.device, non_blocking=True)
                            else:
                                input_ids = batch.to(self.device, non_blocking=True)
                                targets = batch.to(self.device, non_blocking=True)
                        
                        # Forward pass avec optimisations
                        with self.timing_stats.track("forward"), torch.amp.autocast(enabled=True, device_type=self.device_type):
                            # Synchroniser avant le forward pass pour s'assurer que les données sont sur le GPU
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            
                            # Use unified forward pass for all models (same as Lightning)
                            forward_kwargs = {}
                            if model_type == 'llada' and getattr(self.args, 'use_bd3_training', False):
                                forward_kwargs['use_bd3_training'] = True

                            # Call the unified forward method - THIS IS CRITICAL FOR MATCHING LIGHTNING BEHAVIOR
                            outputs = self.forward(input_ids, targets=targets, **forward_kwargs)

                            # Extract outputs from standardized dictionary format
                            logits = outputs.get('logits')
                            loss = outputs.get('loss')
                            router_loss = outputs.get('router_loss')
                            mtp_loss = outputs.get('mtp_loss')
                            ponder_loss = outputs.get('ponder_loss')

                            # Determine balance_loss based on model type
                            balance_loss = 0
                            if model_type == 'deepseek' and mtp_loss is not None:
                                balance_loss = mtp_loss
                            elif router_loss is not None:
                                # Handle router loss with NaN checks
                                if torch.isnan(router_loss).any():
                                    print(f"WARNING: NaN detected in router_loss at iteration {self.iter_num}")
                                    router_loss = torch.where(torch.isnan(router_loss), torch.tensor(0.1, device=router_loss.device), router_loss)
                                router_loss = torch.clamp(router_loss, max=10.0)
                                balance_loss = router_loss

                            # Count non-padding tokens for accuracy (same as Lightning)
                            non_pad_tokens_mask = (targets != -100)
                            batch_tokens = non_pad_tokens_mask.sum().item()

                            # Update token counts
                            self.total_tokens += batch_tokens
                            self.tokens_window.append((time.time(), batch_tokens))
                            if len(self.tokens_window) > self.window_size:
                                self.tokens_window.pop(0)
                            
                            # Clean up to save memory
                            if 'logits' in locals():
                                del logits
                        
                        # Backward pass avec optimisations
                        with self.timing_stats.track("backward"):
                            if loss is not None:
                                # CRITICAL: Save unscaled loss for logging (same as Lightning)
                                unscaled_loss = loss.item()

                                # Scale loss for gradient accumulation
                                loss = loss / self.args.gradient_accumulation_steps

                                # Add auxiliary loss if available
                                if balance_loss != 0:
                                    unscaled_balance_loss = balance_loss.item() if torch.is_tensor(balance_loss) else balance_loss
                                    balance_loss = balance_loss / self.args.gradient_accumulation_steps

                                    # Combine losses (match Lightning logic exactly)
                                    if model_type == 'deepseek' and getattr(self.args, 'use_mtp', True):
                                        # DeepSeek MTP: loss already includes mtp_loss internally, just track it
                                        total_mtp_loss += unscaled_balance_loss
                                        combined_loss = loss  # MTP already combined by model
                                    elif model_type == 'llada':
                                        # Use router loss coefficient for LLaDA
                                        router_loss_coef = getattr(self.args, 'router_z_loss_coef', 0.0001)
                                        total_router_loss += unscaled_balance_loss
                                        combined_loss = loss + router_loss_coef * balance_loss
                                    else:
                                        # Default behavior
                                        router_loss_coef = getattr(self.args, 'router_z_loss_coef', 0.001)
                                        total_router_loss += unscaled_balance_loss
                                        combined_loss = loss + router_loss_coef * balance_loss
                                else:
                                    combined_loss = loss

                                # Check for NaN values before backward pass
                                if torch.isnan(combined_loss).any():
                                    print(f"WARNING: NaN detected in loss at iteration {self.iter_num}")
                                    skip_optimizer_step = True
                                    # Skip backward to avoid corrupting the model
                                    continue

                                # Backward pass with or without scaler
                                if self.scaler.is_enabled():
                                    scaled_loss = self.scaler.scale(combined_loss)
                                    scaled_loss.backward(retain_graph=False)
                                else:
                                    # Direct backward for bfloat16 which has sufficient dynamic range
                                    combined_loss.backward(retain_graph=False)

                                # Track unscaled losses for logging (same as Lightning)
                                total_loss += unscaled_loss
                                
                                # Clean up
                                del loss
                                if balance_loss != 0:
                                    del balance_loss
                                del combined_loss
                                # Supprimer scaled_loss uniquement s'il existe (branche avec scaler)
                                if self.scaler.is_enabled() and 'scaled_loss' in locals():
                                    del scaled_loss
                
                except Exception as e:
                    print(f"Training iteration failed: {e}")
                    print(traceback.format_exc())
                    cleanup_memory()
                    if self.ddp:
                        dist_barrier()
                    continue
                
                # Optimizer step avec optimisations
                with self.timing_stats.track("optimizer_step"):
                    # Skip optimizer step if NaN was detected
                    if skip_optimizer_step:
                        print("Skipping optimizer step due to NaN detected in loss")
                        continue

                    # Calculate gradient norm before clipping (for logging)
                    grad_norm = 0.0
                    if self.args.grad_clip != 0.0:
                        try:
                            # Calculate norm before clipping
                            params_to_clip = self.model.parameters() if not self.ddp else self.model.module.parameters()
                            grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, self.args.grad_clip)
                            grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                        except Exception as e:
                            print(f"Error calculating gradient norm: {e}")
                            grad_norm = 0.0

                    if self.scaler.is_enabled():
                        # Avec GradScaler (pour float16 ou FP8)
                        if self.args.grad_clip != 0.0:
                            self.scaler.unscale_(self.optimizer)
                            # Gradient norm already calculated and clipped above

                        # Étape d'optimisation avec synchronisation
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Sans GradScaler (pour bfloat16 ou float32)
                        # Gradient norm already calculated and clipped above

                        # Check for NaN in gradients before optimizer step
                        has_nan_grad = False
                        for param in (self.model.parameters() if not self.ddp else self.model.module.parameters()):
                            if param.grad is not None and torch.isnan(param.grad).any():
                                has_nan_grad = True
                                break

                        if has_nan_grad:
                            print(f"WARNING: NaN detected in gradients at iteration {self.iter_num}, skipping optimizer step")
                            continue

                        # Étape d'optimisation standard
                        self.optimizer.step()

                    # Synchroniser après l'étape d'optimisation pour maximiser l'utilisation du GPU
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
            
            # Timing and logging
            t1 = time.time()
            dt = t1 - self.t0
            self.t0 = t1
            
            if self.iter_num % self.args.log_interval == 0:
                # Calculate tokens/s for CSV logging
                if len(self.tokens_window) > 1:
                    window_time = self.tokens_window[-1][0] - self.tokens_window[0][0]
                    window_tokens = sum(tokens for _, tokens in self.tokens_window)
                    current_tokens_per_sec = window_tokens / window_time if window_time > 0 else 0
                else:
                    current_tokens_per_sec = 0

                # Calculate average sequence length (excluding padding if possible)
                avg_seq_len = self.args.block_size  # Default to block size

                # Buffer metrics for CSV logging
                self._buffer_metrics_for_csv(total_loss, grad_norm if 'grad_norm' in locals() else 0.0,
                                             current_tokens_per_sec, avg_seq_len, lr)

                # Write CSV periodically (every 10 log intervals)
                if self.iter_num % (self.args.log_interval * 10) == 0:
                    self._log_metrics_to_csv()

                # Choose appropriate auxiliary loss to log based on model type
                if model_type == 'deepseek' and getattr(self.args, 'use_mtp', True):
                    self.log_training_stats(total_loss, total_mtp_loss, dt, lr, loss_type="mtp_loss")
                else:
                    self.log_training_stats(total_loss, total_router_loss, dt, lr, loss_type="router_loss")

                # Save checkpoint periodically
                if self.iter_num % 1000 == 0 and self.master_process:
                    self.save_training_checkpoint()

            # Check for progressive training stage transition
            if self.progressive_training and self.should_progress_to_next_stage():
                if self.master_process:
                    print(f"\nProgressive training: transitioning to next stage at iteration {self.iter_num}")
                self.progress_to_next_stage()

                # Need to recreate data loaders with new block size
                print(f"Recreating data loaders with new block_size={self.args.block_size}")
                self.setup_datasets()

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

        # Flush CSV metrics before exiting
        if self.master_process:
            if self.csv_writer and self.metrics_buffer:
                print("Flushing remaining CSV metrics...")
                self._log_metrics_to_csv()

            # Close CSV file
            if self.csv_file:
                try:
                    self.csv_file.close()
                    print(f"CSV logging closed: {self.csv_file_path}")
                except Exception as e:
                    print(f"Error closing CSV file: {e}")

            # Finish wandb
            if self.use_wandb:
                try:
                    wandb.finish()
                    print("Wandb logging finished")
                except Exception as e:
                    print(f"Error finishing wandb: {e}")

        # Nettoyage final
        cleanup_memory()

        # Arrêter le profiler CUDA si activé
        if torch.cuda.is_available() and hasattr(torch.cuda, 'cudart'):
            torch.cuda.cudart().cudaProfilerStop()

        if self.master_process:
            print("Training finished!")
            
    def get_prompt(self):
        # Get a prompt
        if hasattr(self.args, 'prompt_templates') and self.args.prompt_templates:
            prompt = random.choice(self.args.prompt_templates)
        else:
            # Liste de prompts variés couvrant différents thèmes
            diverse_prompts = [
                # Narration
                "Once upon a time",
                "In a distant world",
                "The story begins with",
                # Questions
                "How can one solve",
                "Why are humans",
                "What is the best way to",
                # Instructions
                "Explain to me how",
                "Write a guide for",
                # Creativity
                "Imagine a scenario where",
                "Describe a futuristic technology that",
                # Analysis
                "Analyze the advantages and disadvantages of",
                "Compare and contrast the following approaches:"
            ]
            prompt = random.choice(diverse_prompts)
            
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
        return prompt, input_tokens
    
    def generate_sample_text(self):
        """Génère un exemple de texte avec le modèle actuel"""
        model_type = self.args.model_type.lower()
        tokenizer = self.args.tokenizer if hasattr(self.args, 'tokenizer') else None
        
        print("\nText Generation:")
        
        try:
            with torch.no_grad(), torch.amp.autocast(enabled=True, device_type=self.device_type):
                # Generate text
                raw_model = self.model.module if self.ddp else self.model
                prompt, input_tokens = self.get_prompt()
                
                # Check if model has a generate method
                if hasattr(raw_model, 'generate'):
                    # Use the model's generate method
                    output = raw_model.generate(
                        input_tokens,
                        max_new_tokens=min(100, self.args.block_size - 10),
                        temperature=0.7,
                        top_k=40
                    )

                    # Handle both tuple and tensor returns
                    if isinstance(output, tuple):
                        output_ids = output[0]  # First element is usually the generated ids
                    else:
                        output_ids = output

                    # Decode the generated text if we have a tokenizer
                    if tokenizer is not None:
                        # Handle batch dimension
                        if isinstance(output_ids, torch.Tensor):
                            if output_ids.dim() > 1:
                                tokens_to_decode = output_ids[0]
                            else:
                                tokens_to_decode = output_ids
                        else:
                            tokens_to_decode = output_ids

                        generated_text = tokenizer.decode(
                            tokens_to_decode,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True
                        )
                        print(f"Generated text: {generated_text}\n")
                    else:
                        if isinstance(output_ids, torch.Tensor):
                            print(f"Generated tokens: {output_ids.tolist()}\n")
                        else:
                            print(f"Generated output: {output_ids}\n")
                    
                else:
                    # Fallback: use the generic generate_text function
                    output_text = generate_text(
                        raw_model,
                        input_tokens,
                        max_new_tokens=min(100, self.args.block_size - 10),
                        temperature=0.7,
                        tokenizer=tokenizer
                    )
                    
                    if output_text is not None:
                        print(f"Generated text: {output_text}\n")
                    else:
                        print(f"Text generation completed (output format depends on model type)\n")
                        
        except Exception as e:
            print(f"Generation error: {str(e)}")
            print(traceback.format_exc())
    
    def evaluate_model(self):
        """Évalue le modèle sur les ensembles d'entraînement et de validation"""
        from train.train_utils import estimate_loss

        print("Validation")
        # Use utility function for loss estimation
        eval_iters = getattr(self.args, 'eval_iters', 200)  # Default to 200 if not specified
        losses = estimate_loss(
            self.model,
            self.train_dataset,
            self.val_dataset,
            eval_iters,
            self.device,
            self.ddp,
            self.ddp_world_size
        )

        print(f"step {self.iter_num}: train loss {losses['train']:.4f}, train ppl {losses['train_ppl']:.2f}, "
              f"val loss {losses['val']:.4f}, val ppl {losses['val_ppl']:.2f}")

        # Store validation metrics for CSV logging
        self.last_val_loss = losses['val']
        self.last_val_perplexity = losses['val_ppl']

        # Log to wandb if enabled
        if self.use_wandb and self.master_process:
            try:
                wandb.log({
                    'val/loss': losses['val'],
                    'val/perplexity': losses['val_ppl'],
                    'train/loss': losses['train'],
                    'train/perplexity': losses['train_ppl']
                }, step=self.iter_num)
            except Exception as e:
                print(f"Warning: Failed to log validation metrics to wandb: {e}")

        # Save checkpoint if best validation loss
        if losses['val'] < self.best_val_loss or self.args.always_save_checkpoint:
            self.best_val_loss = losses['val']
            if self.iter_num > 0:
                self.save_training_checkpoint(val_loss=losses['val'])
    
    def log_training_stats(self, total_loss, aux_loss, dt, lr, loss_type="router_loss"):
        """Affiche les statistiques d'entraînement"""
        lossf = total_loss
        aux_lossf = aux_loss
        
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
        if aux_lossf > 0:
            print(f"iter {self.iter_num}: loss {lossf:.4f}, {loss_type} {aux_lossf:.4f}, "
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
            log_data = {
                "iter": self.iter_num,
                "loss": lossf,
                "tokens_per_sec": current_tokens_per_sec,
                "avg_tokens_per_sec": avg_tokens_per_sec,
                "learning_rate": lr,
                "step_time_ms": dt * 1000
            }
            
            # Add auxiliary loss to logging
            if aux_lossf > 0:
                log_data[loss_type] = aux_lossf
                
            wandb.log(log_data)
    
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