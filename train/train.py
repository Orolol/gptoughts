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
        """Configure l'environnement d'exécution (DDP, device, etc.) avec optimisations avancées"""
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
        else:
            from models.models.model import GPT, GPTConfig
            # Create config based on model size
            config = self.create_gpt_config()
            self.model = GPT(config)
        
        # Store config for checkpointing
        self.config = config
        
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
        
        # Get datasets
        if hasattr(self.args, 'tokenizer') and self.args.tokenizer is not None:
            print(f"Tokenizer found: {self.args.tokenizer}")
            self.train_dataset, self.val_dataset = get_datasets(
                self.args.block_size, 
                self.args.batch_size,
                tokenizer=self.args.tokenizer,
                num_workers=getattr(self.args, 'num_workers', 4)
            )
        else:
            self.train_dataset, self.val_dataset = get_datasets(
                self.args.block_size, 
                self.args.batch_size,
                tokenizer=None,
                num_workers=getattr(self.args, 'num_workers', 4)
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
            if self.iter_num % 500 == 0 and self.master_process or self.iter_num == 50:
            # if self.iter_num 100:
                self.generate_sample_text()
            
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
                            
                            # Handle different model types with unified approach
                            if model_type == 'deepseek':
                                # First try the DeepSeek MTP model return format
                                outputs = self.model(input_ids, targets=targets)
                                
                            elif model_type == 'mla':
                                # MLA model
                                try:
                                    # MLA model returns logits, loss directly
                                    logits, loss = self.model(input_ids, targets=targets)
                                    
                                    # Extract router loss from model if available, but ensure it doesn't have gradients
                                    router_loss = None
                                    if hasattr(self.model, 'last_router_loss') and self.model.last_router_loss is not None:
                                        # last_router_loss should already be a detached tensor with requires_grad=False
                                        router_loss = self.model.last_router_loss
                                        # Create a completely new tensor to ensure no gradient connections
                                        if router_loss is not None:
                                            with torch.no_grad():
                                                router_loss = router_loss.clone()
                                        # Clear the model's router loss after extracting it
                                        self.model.last_router_loss = None
                                    
                                    # Implement additional checks for router_loss
                                    if router_loss is not None:
                                        # Check if router_loss contains NaN values
                                        if torch.isnan(router_loss).any():
                                            print(f"WARNING: NaN detected in router_loss at iteration {self.iter_num}")
                                            # Replace NaN values with a small constant to prevent propagation
                                            router_loss = torch.where(torch.isnan(router_loss), torch.tensor(0.1, device=router_loss.device), router_loss)
                                        
                                        # Cap extremely large router loss values to prevent explosion
                                        router_loss = torch.clamp(router_loss, max=10.0)
                                        balance_loss = router_loss
                                    else:
                                        balance_loss = 0
                                except Exception as e:
                                    print(f"Error during MLA forward pass: {e}")
                                    # Fallback to a simpler pass
                                    logits, loss = self.model(input_ids, targets)
                                    balance_loss = 0
                                    print("Used simplified forward path for MLA model")
                            elif model_type == 'llada':
                                # LLaDA model
                                # Add gradient and loss stabilization for LLaDA models
                                try:
                                    # Apply stabilization techniques specifically to prevent NaN in router mechanism
                                    forward_args = {'input_ids': input_ids, 'targets': targets}
                                    # Add BD3 flag if applicable
                                    # Assumes args object has 'use_bd3_training' attribute (added in run_train.py)
                                    if getattr(self.args, 'use_bd3_training', False):
                                        forward_args['use_bd3_training'] = True
                                    
                                    logits, loss, router_loss = self.model(**forward_args)
                                    
                                    # Implement additional checks for router_loss
                                    if router_loss is not None:
                                        # Check if router_loss contains NaN values
                                        if torch.isnan(router_loss).any():
                                            print(f"WARNING: NaN detected in router_loss at iteration {self.iter_num}")
                                            # Replace NaN values with a small constant to prevent propagation
                                            router_loss = torch.where(torch.isnan(router_loss), torch.tensor(0.1, device=router_loss.device), router_loss)
                                        
                                        # Cap extremely large router loss values to prevent explosion
                                        router_loss = torch.clamp(router_loss, max=10.0)
                                        balance_loss = router_loss
                                    else:
                                        balance_loss = 0
                                except Exception as e:
                                    print(f"Error during LLaDA forward pass: {e}")
                                    # Fallback to a simpler forward pass if there's an error
                                    if hasattr(self.model, 'forward_simple') and callable(getattr(self.model, 'forward_simple')):
                                        logits, loss = self.model.forward_simple(input_ids, targets)
                                        balance_loss = 0
                                        print("Used simplified forward pass to avoid NaN")
                                    else:
                                        raise
                            else:
                                # Standard GPT model
                                logits, loss = self.model(input_ids, targets=targets)
                                balance_loss = 0

                                
                            batch_tokens = input_ids.numel()
                            
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
                                # Scale loss for gradient accumulation
                                loss = loss / self.args.gradient_accumulation_steps
                                
                                # Add auxiliary loss if available
                                if balance_loss != 0:
                                    balance_loss = balance_loss / self.args.gradient_accumulation_steps
                                    
                                    # Track MTP loss separately for logging
                                    if model_type == 'deepseek' and getattr(self.args, 'use_mtp', True):
                                        total_mtp_loss += balance_loss.item()
                                    elif model_type == 'llada':
                                        # Use router loss coefficient for LLaDA
                                        router_loss_coef = getattr(self.args, 'router_z_loss_coef', 0.0001)
                                        total_router_loss += balance_loss.item()
                                        combined_loss = loss + router_loss_coef * balance_loss
                                    else:
                                        # Default behavior
                                        router_loss_coef = getattr(self.args, 'router_z_loss_coef', 0.001)
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
                                    scaled_loss.backward(retain_graph=False)  # Changed to False to reduce memory usage
                                else:
                                    # Direct backward for bfloat16 which has sufficient dynamic range
                                    # Now we can use retain_graph=False for all models including MLA
                                    # since we've fixed the router loss handling
                                    combined_loss.backward(retain_graph=False)
                                    
                                # Track losses
                                total_loss += loss.item()
                                if balance_loss != 0:
                                    total_router_loss += balance_loss.item()
                                
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
                        
                    if self.scaler.is_enabled():
                        # Avec GradScaler (pour float16 ou FP8)
                        if self.args.grad_clip != 0.0:
                            self.scaler.unscale_(self.optimizer)
                            # Apply more aggressive gradient clipping to prevent NaN
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters() if not self.ddp else self.model.module.parameters(),
                                self.args.grad_clip
                            )
                        
                        # Étape d'optimisation avec synchronisation
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Sans GradScaler (pour bfloat16 ou float32)
                        if self.args.grad_clip != 0.0:
                            # Apply gradient clipping to prevent NaN
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters() if not self.ddp else self.model.module.parameters(),
                                self.args.grad_clip
                            )
                        
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
                # Choose appropriate auxiliary loss to log based on model type
                if model_type == 'deepseek' and getattr(self.args, 'use_mtp', True):
                    self.log_training_stats(total_loss, total_mtp_loss, dt, lr, loss_type="mtp_loss")
                else:
                    self.log_training_stats(total_loss, total_router_loss, dt, lr, loss_type="router_loss")
                
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
                    output_ids = raw_model.generate(
                        input_tokens,
                        max_new_tokens=min(100, self.args.block_size - 10),
                        temperature=0.7,
                        top_k=40
                    )
                    
                    # Decode the generated text if we have a tokenizer
                    if tokenizer is not None:
                        generated_text = tokenizer.decode(
                            output_ids[0] if output_ids.dim() > 1 else output_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True
                        )
                        print(f"Generated text: {generated_text}\n")
                    else:
                        print(f"Generated tokens: {output_ids.tolist()}\n")
                    
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
        losses = estimate_loss(
            self.model, 
            self.train_dataset, 
            self.val_dataset, 
            self.args.eval_iters, 
            self.device, 
            self.ddp, 
            self.ddp_world_size
        )
        
        print(f"step {self.iter_num}: train loss {losses['train']:.4f}, train ppl {losses['train_ppl']:.2f}, "
              f"val loss {losses['val']:.4f}, val ppl {losses['val_ppl']:.2f}")
        
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