import os
import sys
import argparse
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, IterableDataset # Import necessary data components

# Conditional imports for Lightning
LIGHTNING_AVAILABLE = False
try:
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
    LIGHTNING_AVAILABLE = True
except (ImportError, RuntimeError):
    pass

# Import local modules
from optimization.cuda_optim import setup_cuda_optimizations, print_gpu_stats
from train.train_utils import find_latest_checkpoint, get_gpu_count

# Set TOKENIZERS_PARALLELISM to avoid warnings with Hugging Face tokenizers when using multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import datasets helper
from data.datasets import get_datasets


print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Vérifier le support de Flash Attention dans PyTorch
print(f"Flash Attention backend: {torch.backends.cuda.flash_sdp_enabled()}")
print(f"Memory efficient attention: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
print(f"Math attention: {torch.backends.cuda.math_sdp_enabled()}")

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM models with PyTorch Lightning')

    # Model Parameters
    parser.add_argument('--model_type', type=str, choices=['deepseek', 'llada', 'sedd', 'gpt', 'mla', 'mla_selective', 'parscale_mla', 'mla_llada', 'mdm', 'moe_mla', 'slm', 'nsa', 'hrm', 'hse', 'swan', 'swa_mla', 'swa_mla_moe'], default='gpt', help='Type of model to train')
    parser.add_argument('--size', type=str, choices=['small', 'medium', 'large', 'xl'], default='small', help='Size of the model')
    parser.add_argument('--use_lightning', action='store_true', default=True, help='Use PyTorch Lightning for training')

    # IO Parameters
    parser.add_argument('--output_dir', type=str, default='ouputs/out_lightning', help='Output directory for checkpoints and logs')
    parser.add_argument('--init_from', type=str, default='scratch', choices=['scratch', 'resume'], help='Initialize from scratch or resume training')
    parser.add_argument('--resume_ckpt_path', type=str, default=None, help='Specific checkpoint path to resume from (overrides searching in output_dir)')
    parser.add_argument('--keep_checkpoints', type=int, default=3, help='Number of checkpoints to keep (-1 for all, 0 to disable checkpointing)')

    # Data Parameters
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size per device')
    parser.add_argument('--block_size', type=int, default=512, help='Context size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of dataloader workers')
    parser.add_argument('--dataloader_type', type=str, default='packed', choices=['original', 'dynamic', 'packed'], help='Type of dataloader to use (`dynamic` is more efficient).')

    # Model Config Parameters (passed to LightningModule)
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing for CE loss (if supported by the model)')
    parser.add_argument('--bias', action='store_true', help='Use bias in linear layers')
    parser.add_argument('--attention_backend', type=str, default=None, help='Attention backend (e.g., flash)')

    # Optimizer Parameters (passed to LightningModule)
    parser.add_argument('--optimizer_type', type=str, default=None, choices=['adamw', 'lion', 'apollo', 'apollo-mini', 'galore', 'galore-8bit', 'galore2'], help='Optimizer type')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.95, help='Adam beta2')
    parser.add_argument('--decay_lr', action='store_true', default=True, help='Decay learning rate') # Default to True as common practice
    parser.add_argument('--warmup_iters', type=int, default=200, help='Warmup iterations')
    parser.add_argument('--lr_decay_iters', type=int, default=600000, help='LR decay iterations')
    parser.add_argument('--min_lr', type=float, default=3e-6, help='Minimum learning rate')

    # Training Parameters (for Lightning Trainer)
    parser.add_argument('--max_iters', type=int, default=100000, help='Maximum training iterations (steps)')
    parser.add_argument('--grad_clip', type=float, default=0.0, help='Gradient clipping value (0 for no clipping)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--eval_interval_steps', type=int, default=10000, help='Validation interval in steps')
    parser.add_argument('--log_interval_steps', type=int, default=1, help='Logging interval in steps')
    parser.add_argument('--eval_only', action='store_true', help='Run only evaluation')
    parser.add_argument('--compile', action='store_true', help='Compile the model with torch.compile')

    # Precision Parameters (for Lightning Trainer)
    parser.add_argument('--precision', type=str, default='bf16-mixed', choices=['32-true', '16-mixed', 'bf16-mixed'], help='Training precision')
    # FP8 requires specific setup, handled separately if needed via transformer_engine integration within the module
    parser.add_argument('--use_fp8', action='store_true', help='Use FP8 precision for models that support it (requires H100/H200 GPU)')
    parser.add_argument('--fp8_mla_params', action='store_true', help='Use FP8 precision for MLA params (default is FP16 for stability)')
    parser.add_argument('--fp8_tile_size', type=int, default=128, help='Tile size for FP8 quantization (default: 128)')

    # Distributed Parameters (for Lightning Trainer)
    parser.add_argument('--strategy', type=str, default='ddp_find_unused_parameters_false',
                       help='Distributed strategy (e.g., ddp, ddp_find_unused_parameters_false, fsdp). Use fsdp for better memory balance.')
    parser.add_argument('--devices', type=int, default=-1, help='Number of GPUs to use (-1 for all available)')
    parser.add_argument('--device', type=str, default=None, help='Device to use for training (e.g., cpu, cuda:0)')
    parser.add_argument('--sync_batchnorm', action='store_true', help='Convert all BatchNorm layers to SyncBatchNorm for multi-GPU')

    # MoE Parameters (passed to LightningModule)
    parser.add_argument('--router_z_loss_coef', type=float, default=0.001, help='Router loss coefficient')
    parser.add_argument('--num_experts', type=int, default=8, help='Number of experts in MoE models')
    parser.add_argument('--experts_per_token', type=int, default=2, help='Number of experts selected per token (top-k routing)')
    parser.add_argument('--shared_weight_ratio', type=float, default=0.75, help='Ratio of weights shared between experts (0.0 to 1.0)')

    # HSE-specific Parameters
    parser.add_argument('--scribe_chunk_size', type=int, default=2048, help='Scribes local chunk size')
    parser.add_argument('--scribe_summary_len', type=int, default=128, help='Number of summary tokens produced by Scribes')
    parser.add_argument('--qap_per_step', type=int, default=12, help='QAP budget per decoding step')
    parser.add_argument('--qap_per_expert', type=int, default=6, help='QAP budget per expert per step')
    parser.add_argument('--qap_max_queries', type=int, default=20, help='Max QAP queries per step')
    parser.add_argument('--ratio_kv', type=int, default=8, help='KV head ratio for GQA in standard attention')
    # SWAN architecture overrides
    parser.add_argument('--n_layer', type=int, default=None, help='Override number of transformer layers for SWAN models')
    parser.add_argument('--n_head', type=int, default=None, help='Override number of attention heads for SWAN models')
    parser.add_argument('--n_embd', type=int, default=None, help='Override embedding dimension for SWAN models')
    parser.add_argument('--global_layers_per_cycle', type=int, default=None, help='Number of global NoPE layers per SWAN cycle')
    parser.add_argument('--local_layers_per_cycle', type=int, default=None, help='Number of SWA-RoPE layers per SWAN cycle')
    parser.add_argument('--swa_window', type=int, default=None, help='Sliding window size for SWA-RoPE layers in SWAN')
    parser.add_argument('--logit_scale_base', type=float, default=None, help='Base parameter a for SWAN logit scaling log_a(a+n)')
    parser.add_argument('--logit_scale_window', type=int, default=None, help='Token window granularity for SWAN logit scaling')
    parser.add_argument('--logit_scale_offset', type=int, default=None, help='Offset applied before SWAN logit scaling windowing')
    parser.add_argument('--logit_scale_min', type=float, default=None, help='Minimum SWAN logit scaling factor')
    parser.add_argument('--logit_scale_max', type=float, default=None, help='Maximum SWAN logit scaling factor')
    parser.add_argument('--swa_layers_per_cycle', type=int, default=None, help='Number of SWA layers per cycle for SWA+MLA hybrid')
    parser.add_argument('--mla_layers_per_cycle', type=int, default=None, help='Number of MLA layers per cycle for SWA+MLA hybrid')
    parser.add_argument('--mla_q_lora_rank', type=int, default=0, help='Q projection LoRA rank for MLA blocks')
    parser.add_argument('--mla_kv_lora_rank', type=int, default=512, help='KV projection LoRA rank for MLA blocks')
    parser.add_argument('--mla_qk_nope_head_dim', type=int, default=128, help='NoPE head dim for MLA blocks')
    parser.add_argument('--mla_qk_rope_head_dim', type=int, default=64, help='RoPE head dim for MLA blocks')
    parser.add_argument('--mla_v_head_dim', type=int, default=128, help='Value head dim for MLA blocks')
    parser.add_argument('--mla_attn_impl', type=str, default='absorb', help='Attention implementation for MLA blocks')
    parser.add_argument('--mla_rope_factor', type=float, default=1.0, help='RoPE factor for MLA scaling')
    parser.add_argument('--mla_mscale', type=float, default=1.0, help='MSCALE factor for MLA extended contexts')
    parser.add_argument('--use_mla_selective', action='store_true', help='Use MLA Selective instead of standard MLA in SWA-MLA hybrid')
    parser.add_argument('--mla_selection_head_idx', type=int, default=0, help='Which attention head to use for selection in MLA Selective')
    parser.add_argument('--swa_sink_size', type=int, default=4, help='Number of initial tokens for attention sink in SWA blocks')

    # BD3-LM Specific Args (passed to LightningModule)
    parser.add_argument('--use_bd3_training', action='store_true', help='Enable BD3-LM vectorized training path for LLaDA model')
    parser.add_argument('--bd3_block_length', type=int, default=128, help='Block length for BD3 training')
    parser.add_argument('--bd3_beta', type=float, default=0.3, help='Minimum masking rate for BD3 clipped schedule')
    parser.add_argument('--bd3_omega', type=float, default=0.8, help='Maximum masking rate for BD3 clipped schedule')
    parser.add_argument('--disable_entropy_regularization', action='store_true', help='Disable entropy regularization in LLaDA loss')

    # Tokenizer Parameters
    parser.add_argument('--tokenizer_name', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help='Tokenizer name from Hugging Face Hub')

    # Logging
    parser.add_argument('--wandb_project', type=str, default=None, help='WandB project name (if None, uses TensorBoard)')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity name')

    # Advanced Optimizations (passed to LightningModule)
    parser.add_argument('--optimize_attention', action='store_true', help='Enable attention optimizations (if available)')
    parser.add_argument('--preallocate_memory', action='store_true', help='Preallocate CUDA memory (if available)')
    
    # Profiling Parameters
    parser.add_argument('--profile', action='store_true', help='Enable profiling for NSA model')
    parser.add_argument('--profile_interval', type=int, default=100, help='Steps between profiling summaries')
    
    # Dynamic Tanh (DyT) Parameters
    parser.add_argument('--use_dyt', action='store_true', help='Use Dynamic Tanh (DyT) instead of RMSNorm for ~8% speedup')
    parser.add_argument('--dyt_alpha_init', type=float, default=0.5, help='Initial value for DyT alpha parameter')

    # HRM-specific Parameters
    parser.add_argument('--hrm_max_segments', type=int, default=None, help='Override: maximum number of HRM segments (ACT)')
    parser.add_argument('--hrm_cycles_per_segment', type=int, default=None, help='Override: number of cycles per HRM segment (N)')
    parser.add_argument('--hrm_steps_per_cycle', type=int, default=None, help='Override: number of L-steps per cycle (T)')
    parser.add_argument('--ponder_loss_weight', type=float, default=0.01, help='HRM ponder loss weight')
    parser.add_argument('--halt_bias_init', type=float, default=-2.0, help='Initial bias for HRM halting head (encourages early halting)')
    parser.add_argument('--hrm_deq_one_step', action='store_true', help='Enable DEQ-style 1-step gradient (memory O(1) within segment)')
    parser.add_argument('--hrm_use_deep_supervision', action='store_true', help='Enable deep supervision across segments with detach between segments')
    parser.add_argument('--hrm_n_supervision_segments', type=int, default=1, help='Number of supervision segments if deep supervision is enabled')
    parser.add_argument('--hrm_gradient_steps', type=int, default=None, help='Number of steps with gradients (-1 for all, 1 for 1-step approx)')
    parser.add_argument('--hrm_use_act', type=str, default=None, help='Enable/disable Adaptive Computation Time (true/false)')

    # ParScale-MLA Parameters
    parser.add_argument('--parallel_streams', type=int, default=8, help='Number of parallel streams for ParScale')
    parser.add_argument('--prefix_length', type=int, default=48, help='Length of input-space prefixes')
    parser.add_argument('--latent_prefix_length', type=int, default=16, help='Length of latent-space prefixes')
    parser.add_argument('--aggregator_epsilon', type=float, default=0.1, help='Label smoothing for aggregation')
    parser.add_argument('--diversity_weight', type=float, default=0.1, help='Weight for diversity regularization')
    parser.add_argument('--use_dynamic_inference', action='store_true', default=True, help='Enable dynamic inference based on complexity')
    parser.add_argument('--complexity_threshold', type=float, default=0.5, help='Threshold for full stream activation')
    parser.add_argument('--training_stage', type=int, default=1, choices=[1, 2], help='ParScale training stage (1: base, 2: parscale)')
    parser.add_argument('--freeze_base_in_stage2', action='store_true', default=True, help='Freeze base model in stage 2')
    parser.add_argument('--base_checkpoint', type=str, default=None, help='Base model checkpoint for ParScale stage 2')
    
    # GaLore Parameters
    parser.add_argument('--galore_rank', type=int, default=128, help='GaLore low-rank dimension')
    parser.add_argument('--galore_update_proj_gap', type=int, default=200, help='GaLore projection update interval')
    parser.add_argument('--galore_scale', type=float, default=0.25, help='GaLore scaling factor')
    parser.add_argument('--galore_proj_type', type=str, default='std', help='GaLore projection type')
    parser.add_argument('--galore_quantize_proj', type=int, default=None, help='GaLore2 projection quantization (1 or 2 bits)')
    
    # Selective Attention Parameters (for MLA-Selective model)
    parser.add_argument('--selection_ratio', type=float, default=0.5, help='Ratio of tokens to select (0.0 to 1.0)')
    parser.add_argument('--selection_method', type=str, default='top_k', choices=['top_k', 'threshold', 'gumbel'], help='Method for token selection')
    parser.add_argument('--selection_temperature', type=float, default=1.0, help='Temperature for Gumbel selection')
    
    # MLA-LLaDA Parameters
    parser.add_argument('--remasking_strategy', type=str, default='low_confidence', choices=['low_confidence', 'random'], help='Remasking strategy for LLaDA diffusion')
    parser.add_argument('--num_diffusion_steps', type=int, default=None, help='Number of diffusion steps for generation (None for adaptive)')
    parser.add_argument('--mask_ratio_min', type=float, default=0.15, help='Minimum masking ratio for training')
    parser.add_argument('--mask_ratio_max', type=float, default=0.85, help='Maximum masking ratio for training')

    args = parser.parse_args()
    return args

# --- Main Execution ---
def main():
    args = parse_args()

    # Setup CUDA optimizations early
    if torch.cuda.is_available():
        setup_cuda_optimizations()
        if args.devices == -1: # Auto-detect GPUs if not specified
             args.devices = get_gpu_count()
        print(f"Detected {args.devices} GPUs.")
        if args.devices > 0:
             print_gpu_stats() # Print initial stats

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Tokenizer ---
    try:
        access_token = os.getenv('HF_TOKEN')
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True, access_token=access_token)
        tokenizer.pad_token = tokenizer.eos_token # Set pad token if needed
        args.vocab_size = len(tokenizer) # Get vocab size from len(tokenizer) for robustness
        args.tokenizer = tokenizer # Pass tokenizer via args for generation in module
        print(f"Initialized tokenizer: {args.tokenizer_name} with effective vocab size {args.vocab_size} (from len(tokenizer))")
    except Exception as e:
        print(f"Failed to load tokenizer '{args.tokenizer_name}': {e}")
        print("Proceeding without a tokenizer. Generation and some datasets might not work.")
        # Estimate vocab size or set a default if needed by model config
        args.vocab_size = getattr(args, 'vocab_size', 32000) # Use provided or default
        args.tokenizer = None

    # --- DataLoaders ---
    print("Setting up datasets...")
    train_loader, val_loader = get_datasets(args)
    print("Datasets ready.")

    # Choose between Lightning and standard training
    if args.use_lightning and LIGHTNING_AVAILABLE:
        print("Using PyTorch Lightning for training...")
        
        # Import Lightning module if available
        from train.lightning_module import LLMLightningModule
        
        # --- LightningModule ---
        print("Initializing LightningModule...")
        model = LLMLightningModule(args)
        print("LightningModule initialized.")
    
        # --- Callbacks ---
        callbacks = []
        
        # Only add checkpoint callback if checkpointing is enabled
        if args.keep_checkpoints != 0:
            checkpoint_callback = ModelCheckpoint(
                dirpath=args.output_dir,
                filename='{epoch}-{step}-{val/loss:.2f}',
                save_top_k=args.keep_checkpoints,
                monitor='val/loss',
                mode='min',
                save_last=True, # Always save the last checkpoint
                every_n_train_steps=args.eval_interval_steps # Save checkpoint after validation
            )
            callbacks.append(checkpoint_callback)
        else:
            print("Checkpointing disabled (keep_checkpoints=0)")
        
        lr_monitor = LearningRateMonitor(logging_interval='step')
        progress_bar = TQDMProgressBar(refresh_rate=10) # Adjust refresh rate as needed
        
        callbacks.extend([lr_monitor, progress_bar])
    
        # --- Logger ---
        if args.wandb_project:
            logger = WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                log_model=False, # Don't log model checkpoints to WandB by default
                save_dir=args.output_dir,
                config=vars(args) # Log hyperparameters
            )
            print(f"Using WandB logger (Project: {args.wandb_project})")
        else:
            logger = TensorBoardLogger(
                save_dir=args.output_dir,
                name="logs"
            )
            print(f"Using TensorBoard logger (Directory: {args.output_dir}/logs)")
    
        # --- Trainer ---
        # Determine checkpoint path for resuming
        ckpt_path = None
        if args.init_from == 'resume':
            if args.resume_ckpt_path:
                ckpt_path = args.resume_ckpt_path
                print(f"Resuming from specified checkpoint: {ckpt_path}")
            else:
                # Try to find the last checkpoint in the output directory
                # Check for PyTorch Lightning checkpoint first
                pl_ckpt = os.path.join(args.output_dir, "last.ckpt")
                if os.path.exists(pl_ckpt):
                    ckpt_path = pl_ckpt
                else:
                    # Fallback to custom checkpoint format
                    ckpt_path = find_latest_checkpoint(args.output_dir)
                if ckpt_path:
                    print(f"Resuming from last checkpoint found: {ckpt_path}")
                else:
                    print(f"Resume requested but no checkpoint found in {args.output_dir}. Starting from scratch.")
                    args.init_from = 'scratch' # Fallback to scratch if no checkpoint
    
        # Configure Trainer with multi-GPU optimizations
        from pytorch_lightning.strategies import DDPStrategy
        try:
            from pytorch_lightning.strategies import FSDPStrategy
            fsdp_available = True
        except ImportError:
            fsdp_available = False

        # Configure strategy for multi-GPU
        if args.devices > 1:
            if 'fsdp' in args.strategy.lower():
                if not fsdp_available:
                    print("WARNING: FSDP requested but not available. Falling back to DDP.")
                    args.strategy = 'ddp_find_unused_parameters_false'
                else:
                    print(f"Configuring FSDPStrategy for better memory distribution...")
                    print(f"  - FSDP shards model parameters across GPUs")
                    print(f"  - This should eliminate VRAM imbalance between ranks")

                    # Custom wrapping policy: don't wrap embedding/lm_head that share weights
                    from functools import partial
                    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

                    # Import the transformer block classes that should be wrapped
                    try:
                        from models.models.swa_mla_model import SWALocalBlock
                        from models.blocks.mla_block import MLABlock
                        transformer_layer_cls = {SWALocalBlock, MLABlock}
                    except:
                        # Fallback if imports fail
                        transformer_layer_cls = {torch.nn.TransformerEncoderLayer, torch.nn.TransformerDecoderLayer}

                    my_auto_wrap_policy = partial(
                        transformer_auto_wrap_policy,
                        transformer_layer_cls=transformer_layer_cls,
                    )

                    strategy = FSDPStrategy(
                        auto_wrap_policy=my_auto_wrap_policy,
                        activation_checkpointing_policy=None,
                        state_dict_type="full",
                    )

            if 'ddp' in args.strategy.lower():
                ddp_kwargs = {
                    'find_unused_parameters': False,  # Performance optimization
                    'gradient_as_bucket_view': True,  # Memory optimization
                    'static_graph': True,  # Faster for models with static computation graphs
                    'broadcast_buffers': False,  # Prevent buffer duplication on rank 1
                    'bucket_cap_mb': 10,  # Reduce from default 25MB to balance VRAM reserved between ranks
                }

                # Only add find_unused_parameters if explicitly requested
                if args.strategy == 'ddp':
                    ddp_kwargs['find_unused_parameters'] = True
                    ddp_kwargs['static_graph'] = False

                print(f"Configuring DDPStrategy with optimizations for memory balance...")
                print(f"  - broadcast_buffers=False")
                print(f"  - gradient_as_bucket_view=True")
                print(f"  - static_graph={ddp_kwargs['static_graph']}")
                print(f"  - bucket_cap_mb={ddp_kwargs['bucket_cap_mb']} (reduced from 25MB default)")

                strategy = DDPStrategy(**ddp_kwargs)
            elif 'fsdp' not in args.strategy.lower():
                strategy = args.strategy
        else:
            strategy = "auto"

        trainer_kwargs = {
            'devices': args.devices,
            'accelerator': "gpu" if torch.cuda.is_available() and args.devices != 0 else "cpu",
            'strategy': strategy,
            'precision': args.precision,
            'max_steps': args.max_iters,
            'val_check_interval': args.eval_interval_steps,
            'check_val_every_n_epoch': None,
            'log_every_n_steps': args.log_interval_steps,
            'accumulate_grad_batches': args.gradient_accumulation_steps,
            'gradient_clip_val': args.grad_clip if args.grad_clip > 0 else 0,
            'logger': logger,
            'callbacks': callbacks,
            'enable_checkpointing': (args.keep_checkpoints != 0),
            'benchmark': True,
            'limit_val_batches': 50,
            'use_distributed_sampler': False
        }

        # Add sync_batchnorm for multi-GPU if requested
        if args.devices > 1 and getattr(args, 'sync_batchnorm', False):
            trainer_kwargs['sync_batchnorm'] = True
            print("Enabling SyncBatchNorm for multi-GPU training")

        trainer = pl.Trainer(**trainer_kwargs)
    
        # --- Start Training with Lightning ---
        if args.eval_only:
            print("Starting evaluation only...")
            if not ckpt_path:
                 print("Error: Evaluation only requested but no checkpoint specified or found.")
                 sys.exit(1)
            trainer.validate(model, dataloaders=val_loader, ckpt_path=ckpt_path)
            print("Evaluation finished.")
        else:
            print("Starting training with Lightning...")
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
            print("Lightning training finished.")
    
    else:
        # Lightning not available or not requested
        if args.use_lightning and not LIGHTNING_AVAILABLE:
            print("PyTorch Lightning is not available. Falling back to standard training.")
        else:
            print("Using standard training (non-Lightning)...")
        
        # Import Trainer from train.py
        from train.train import Trainer
        
        # Set evaluation interval for standard training
        args.eval_interval = args.eval_interval_steps  # Rename to match standard training parameter name
        args.log_interval = args.log_interval_steps    # Rename to match standard training parameter name
        
        # Create and run trainer
        trainer = Trainer(args)
        
        if args.eval_only:
            print("Evaluation only mode not supported yet in standard training. Use Lightning for evaluation.")
            sys.exit(1)
        else:
            print("Starting standard training...")
            trainer.train()
            print("Standard training finished.")

if __name__ == "__main__":
    main()
