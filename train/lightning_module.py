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
from datetime import datetime

# Import necessary components from your project
from models.deepseek.deepseek_adapter_mtp import DeepSeekMiniMTP, DeepSeekMiniConfigMTP
from models.llada.model import LLaDAModel
from models.models.model import GPT
from models.models.mla_model import MLAModel, MLAModelConfig
from models.models.parscale_mla import ParScaleMLA, ParScaleMLAConfig, create_parscale_mla
from models.models.mla_selective_model import MLASelectiveModel, MLASelectiveModelConfig
from models.models.moe_mla_model import MOEMLA, MOEMLAConfig
from models.mdm.model import MDMModel
from models.config import MDMConfig
from train.train_utils import (
    get_lr, calculate_perplexity, ensure_model_dtype,
    AveragedTimingStats, generate_text, estimate_loss
)
from optimization.memory_optim import cleanup_memory, print_memory_stats, preallocate_cuda_memory
from optimization.cuda_optim import setup_cuda_optimizations, print_gpu_stats
from optimization.training_optim import enable_torch_compile
from optimization.fp8_deepseek_trainer import FP8AdamW, FP8MixedPrecisionTrainer
from models.galore2_fixed import GaLore2AdamW


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
            if hasattr(self.args, 'preallocate_memory') and self.args.preallocate_memory:
                preallocate_cuda_memory()
            if self.global_rank == 0:
                 print_gpu_stats()
        
        # Compile model if requested
        if hasattr(self.args, 'compile') and self.args.compile:
            # Skip compilation for models with known torch.compile compatibility issues
            skip_compile_models = ['mla_llada']
            if self.args.model_type.lower() in skip_compile_models:
                print(f"Skipping model compilation for {self.args.model_type} model (torch.compile compatibility issues with gradient checkpointing)")
            else:
                # For MLA-selective model, disable gradient checkpointing before compilation
                if self.args.model_type.lower() == 'mla_selective':
                    print("Disabling gradient checkpointing for MLA-selective model compilation...")
                    for module in self.model.modules():
                        if hasattr(module, 'use_checkpoint'):
                            module.use_checkpoint = False
                
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
            n_head = 24
        elif self.args.size == 'large':
            n_layer = 32
            n_embd = 2048
            n_head = 32
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
            intermediate_size = 3072  # Increased proportionally
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
            num_experts=getattr(self.args, 'num_experts', 8),
            experts_per_token=getattr(self.args, 'experts_per_token', 2),
            shared_weight_ratio=getattr(self.args, 'shared_weight_ratio', 0.75),
            
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
        elif model_type == 'mla_selective':
            # MLA-Selective Model returns logits, loss directly
            logits, loss = self.model(input_ids, targets)
            return logits, loss, None
        elif model_type == 'parscale_mla':
            # ParScale-MLA returns logits, loss directly
            logits, loss = self.model(input_ids, targets)
            return logits, loss, None
        elif model_type == 'mla_llada':
            # MLA-LLaDA returns dict with loss/logits
            try:
                output = self.model(input_ids, labels=targets, is_training=True)
                # Ensure we have valid outputs
                if isinstance(output, dict):
                    logits = output.get('logits')
                    loss = output.get('loss')
                    if logits is None or loss is None:
                        raise ValueError(f"MLA-LLaDA model returned incomplete output: {output.keys()}")
                    # Log mask ratio if available
                    if 'mask_ratio' in output and hasattr(self, 'log'):
                        mask_ratio = output['mask_ratio']
                        if torch.is_tensor(mask_ratio):
                            self.log('train/mask_ratio', float(mask_ratio.item()), on_step=True, on_epoch=False, sync_dist=True)
                    return logits, loss, None
                else:
                    raise ValueError(f"MLA-LLaDA model returned unexpected output type: {type(output)}")
            except Exception as e:
                print(f"Error in MLA-LLaDA forward: {e}")
                print(f"Input shapes - input_ids: {input_ids.shape}, targets: {targets.shape if targets is not None else None}")
                raise
        elif model_type == 'mdm':
            # MDM returns dict with loss/logits
            output = self.model(input_ids, labels=targets, return_dict=True)
            logits = output['logits']
            loss = output['loss']
            # Log masking information if available
            if 'mask' in output and hasattr(self, 'log'):
                mask_ratio = output['mask'].float().mean()
                self.log('train/mask_ratio', mask_ratio.item(), on_step=True, on_epoch=False, sync_dist=True)
            return logits, loss, None
        elif model_type == 'moe_mla':
            # MOE-MLA returns logits, loss directly with router loss incorporated
            logits, loss = self.model(input_ids, targets)
            # Router loss is already included in the loss
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
                
                # Add special handling for MLA and ParScale-MLA models
                if self.args.model_type.lower() in ['mla', 'mla_selective', 'parscale_mla', 'mla_llada', 'moe_mla']:
                    try:
                        # Ensure the model knows we're in training mode
                        self.model.train()
                        
                        # Explicitly reset any inference-mode caches
                        if hasattr(self.model, '_set_inference_mode'):
                            self.model._set_inference_mode(False)
                            
                        # Run the forward pass
                        logits, loss, router_loss = self(input_ids, targets=targets)
                        
                        # Ensure the loss has requires_grad for optimizer
                        if loss is not None and not loss.requires_grad:
                            loss = loss.clone().requires_grad_(True)
                    except Exception as e:
                        print(f"Error in MLA/ParScale-MLA forward pass: {e}")
                        # Create error tensor with gradient for safe fallback
                        logits = torch.zeros(1, device=self.device).requires_grad_(True)
                        loss = torch.tensor(10.0, device=self.device).requires_grad_(True)
                        router_loss = None
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

                # For MLA, ParScale-MLA, and MOE-MLA models, router loss is already incorporated internally
                model_type = getattr(self.args, 'model_type', 'gpt')
                
                if model_type in ['mla', 'parscale_mla', 'moe_mla']:
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
        loss_value = float(loss.item())
        self.log('train/loss', loss_value, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        
        # Calculate gradient norms for monitoring
        grad_norm = 0.0
        if self.global_step % 10 == 0:  # Calculate grad norm every 10 steps
            with torch.no_grad():
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                grad_norm = total_norm ** 0.5
                self.log('train/grad_norm', grad_norm, on_step=True, on_epoch=False, sync_dist=True)
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
        
        # Calculate tokens per second per million parameters
        param_count_millions = sum(p.numel() for p in self.model.parameters()) / 1e6
        if param_count_millions > 0:
            tokens_per_sec_per_M_params = current_tokens_per_sec / param_count_millions
            avg_tokens_per_sec_per_M_params = avg_tokens_per_sec / param_count_millions
            self.log('tokens_per_sec_per_M_params', tokens_per_sec_per_M_params, on_step=True, on_epoch=False, prog_bar=True, sync_dist=False)
            self.log('tokens_per_sec_per_M_params_avg', avg_tokens_per_sec_per_M_params, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        
        # Store metrics in buffer for CSV logging
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get validation metrics from callback metrics (if available)
        # Update last known values if new ones are available
        current_val_loss = self.trainer.callback_metrics.get('val/loss', None)
        current_val_perplexity = self.trainer.callback_metrics.get('val/perplexity', None)
        
        if current_val_loss is not None:
            self.last_val_loss = current_val_loss
        if current_val_perplexity is not None:
            self.last_val_perplexity = current_val_perplexity
        
        # Use the last known values for logging
        val_loss = self.last_val_loss
        val_perplexity = self.last_val_perplexity
        
        # Calculate tokens per second per million parameters for CSV
        param_count_millions = sum(p.numel() for p in self.model.parameters()) / 1e6
        tokens_per_sec_per_M_params = current_tokens_per_sec / param_count_millions if param_count_millions > 0 else 0
        
        self.metrics_buffer.append([
            self.global_step,
            f"{loss_value:.6f}",
            f"{val_loss:.6f}" if val_loss is not None else "N/A",
            f"{val_perplexity:.6f}" if val_perplexity is not None else "N/A",
            f"{lr:.2e}",
            f"{current_tokens_per_sec:.2f}",
            f"{tokens_per_sec_per_M_params:.2f}",
            f"{self.total_tokens}",
            f"{grad_norm:.4f}" if grad_norm > 0 else "N/A",
            timestamp
        ])

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

        elif self.global_step % 100 == 0:
            self.generate_sample_text()
            
            # Log ParScale-specific metrics if applicable
            if self.args.model_type == 'parscale_mla' and hasattr(self.model, 'analyze_stream_diversity'):
                diversity_stats = self.model.analyze_stream_diversity()
                self.log('parscale/input_similarity', diversity_stats['input_similarity'], on_step=True, on_epoch=False, sync_dist=True)
                self.log('parscale/layer_similarity', diversity_stats['layer_similarity'], on_step=True, on_epoch=False, sync_dist=True)
                self.log('parscale/effective_streams', diversity_stats['effective_streams'], on_step=True, on_epoch=False, sync_dist=True)
            
            # Log metrics to CSV every 100 steps
            self._log_metrics_to_csv()
            
            # cleanup_memory()
        
        # I want to print logs to check what tokens are sent to the models periodically
        # if self.global_step % 100 == 0:
        #     print(f"TEXT INPUT: {self.args.tokenizer.decode(input_ids[0])}")
        #     print(f"TEXT TARGET: {self.args.tokenizer.decode(targets[0])}")

        # Clear MLA caches after each training step to prevent memory accumulation
        if hasattr(self.model, 'clear_cache') and self.args.model_type.lower() in ['mla', 'parscale_mla', 'mla_selective', 'moe_mla']:
            self.model.clear_cache()
        # Clear diffusion cache for MLA-LLaDA
        elif self.args.model_type.lower() == 'mla_llada' and hasattr(self.model, 'cache_manager'):
            self.model.cache_manager.clear()
        
        return combined_loss

    def validation_step(self, batch, batch_idx):
        input_ids, targets = self._unpack_batch(batch)

        # Forward pass - detach inputs just to be safe
        with torch.no_grad():
            input_ids = input_ids.detach().clone().requires_grad_(False)
            targets = targets.detach().clone().requires_grad_(False)
            
        # Forward pass with model-specific handling
        if self.args.model_type.lower() in ['mla', 'parscale_mla', 'mla_llada', 'moe_mla']:
            try:
                # Ensure the model knows we're in eval mode
                self.model.eval()
                
                # Run the forward pass
                logits, loss, router_loss = self(input_ids, targets=targets)
                
                # For validation, we don't need gradients
                if loss is not None:
                    loss = loss.detach()
            except Exception as e:
                print(f"Error in MLA/ParScale-MLA validation: {e}")
                # Create dummy values for safe fallback
                logits = torch.zeros(1, device=self.device)
                loss = torch.tensor(10.0, device=self.device)
                router_loss = None
        else:
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
        # if self.global_rank == 0 and hasattr(self.args, 'tokenizer') and self.args.tokenizer:
        #     self.generate_sample_text()

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
        
        # Create directory for logs if it doesn't exist
        log_dir = os.path.join(getattr(self.args, 'out_dir', 'out'), 'metrics_logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Find a unique filename by adding timestamp and/or counter
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{base_filename}_{timestamp}.csv"
        self.csv_file_path = os.path.join(log_dir, csv_filename)
        
        # If file exists, add a counter
        counter = 1
        while os.path.exists(self.csv_file_path):
            csv_filename = f"{base_filename}_{timestamp}_{counter}.csv"
            self.csv_file_path = os.path.join(log_dir, csv_filename)
            counter += 1
        
        # Open CSV file and write headers
        self.csv_file = open(self.csv_file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['step', 'train_loss', 'val_loss', 'val_perplexity', 'learning_rate', 'tokens_per_sec', 'tokens_per_sec_per_M_params', 'total_tokens', 'grad_norm', 'timestamp'])
        self.csv_file.flush()
        
        print(f"CSV metrics logging initialized: {self.csv_file_path}")
    
    def _log_metrics_to_csv(self):
        """Write buffered metrics to CSV file every 100 steps."""
        if self.global_step % 100 != 0 or not self.csv_writer or not self.metrics_buffer:
            return
            
        # Count rows before clearing
        num_rows = len(self.metrics_buffer)
        
        # Write all buffered rows to CSV
        for row in self.metrics_buffer:
            self.csv_writer.writerow(row)
        
        # Flush to disk
        self.csv_file.flush()
        
        # Clear buffer for next window
        self.metrics_buffer = []
        
        if self.global_rank == 0:
            print(f"Written {num_rows} rows to CSV (up to step {self.global_step})")

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
             
             # Close CSV file
             if self.csv_file:
                 self.csv_file.close()
                 print(f"CSV logging closed: {self.csv_file_path}")
             
             if self.global_rank == 0:
                 print("Training finished. Final memory stats:")
                 print_memory_stats("Teardown")


# Note: DataModule definition would go here or in a separate file.
# For now, we assume dataloaders are passed directly to trainer.fit()