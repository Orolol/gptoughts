"""
Optimisations spécifiques pour RTX 5090 (Ada Lovelace, 32GB VRAM)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import os

class RTX5090Optimizer:
    """Optimiseur de configuration pour RTX 5090"""
    
    @staticmethod
    def get_optimal_config(model_type: str = "mla", size: str = "large") -> Dict[str, Any]:
        """
        Retourne la configuration optimale pour RTX 5090
        
        Args:
            model_type: Type de modèle (mla, nsa, hrm, etc.)
            size: Taille du modèle (small, medium, large, xl)
        
        Returns:
            Configuration optimisée
        """
        
        # Configuration de base pour 32GB VRAM
        base_config = {
            # Batch sizes optimaux pour RTX 5090
            "batch_size": {
                "small": 64,
                "medium": 48,
                "large": 32,
                "xl": 16
            }[size],
            
            # Longueur de séquence maximale
            "block_size": {
                "small": 8192,
                "medium": 6144,
                "large": 4096,
                "xl": 2048
            }[size],
            
            # Gradient accumulation
            "gradient_accumulation_steps": {
                "small": 1,
                "medium": 2,
                "large": 2,
                "xl": 4
            }[size],
            
            # Learning rate optimisé pour batch size élevé
            "learning_rate": {
                "small": 8e-4,
                "medium": 6e-4,
                "large": 4e-4,
                "xl": 3e-4
            }[size],
            
            # Paramètres AdamW optimisés
            "beta1": 0.9,
            "beta2": 0.95,  # Plus stable pour gros batch
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            
            # Warmup plus long pour gros batch
            "warmup_iters": 2000,
            "lr_decay_iters": 50000,
            "min_lr": 6e-5,
            
            # Précision mixte
            "dtype": "bfloat16",  # BF16 pour Ada Lovelace
            "use_fp8": True,  # FP8 supporté sur Ada
            "fp8_tile_size": 256,  # Optimisé pour tensor cores
            
            # Optimisations mémoire
            "gradient_checkpointing": size in ["large", "xl"],
            "optimize_attention": True,
            "attention_backend": "flash",  # Flash Attention 2
            "preallocate_memory": True,
            
            # Data loading
            "num_workers": 8,
            "prefetch_factor": 4,
            "persistent_workers": True,
            
            # Compilation
            "compile": True,
            "compile_mode": "max-autotune",
        }
        
        # Ajustements spécifiques par modèle
        if model_type == "mla":
            base_config.update({
                "kv_lora_rank": {
                    "small": 256,
                    "medium": 512,
                    "large": 768,
                    "xl": 1024
                }[size],
                "use_moe": False,  # MoE consomme beaucoup de VRAM
            })
        elif model_type == "nsa":
            base_config.update({
                "compress_block_size": 64,
                "compress_stride": 32,
                "selection_block_size": 128,
                "num_selected_blocks": 20,
                "sliding_window_size": 512,
            })
        elif model_type == "hrm":
            base_config.update({
                "hrm_cycles_per_segment": 4,
                "hrm_steps_per_cycle": 4,
                "hrm_max_segments": 12,
                "hrm_use_act": True,
            })
        
        return base_config
    
    @staticmethod
    def setup_cuda_optimizations():
        """Configure les optimisations CUDA pour RTX 5090"""
        
        # Tensor Cores Ada Lovelace
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Flash SDP pour attention
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)  # Désactiver pour forcer Flash/Mem-eff
        
        # Optimisations cuDNN
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Cache et allocation mémoire
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.95)  # Utiliser 95% de la VRAM
        
        # Caching allocator optimisé
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
        
        print("✅ RTX 5090 CUDA optimizations configured")
    
    @staticmethod
    def optimize_model_for_5090(model: nn.Module) -> nn.Module:
        """
        Optimise un modèle pour RTX 5090
        
        Args:
            model: Le modèle PyTorch à optimiser
            
        Returns:
            Modèle optimisé
        """
        
        # Activation checkpointing sélectif
        def selective_checkpoint(module):
            """Active le checkpointing seulement pour les couches gourmandes"""
            if isinstance(module, (nn.MultiheadAttention, nn.TransformerEncoderLayer)):
                if hasattr(module, 'gradient_checkpointing_enable'):
                    module.gradient_checkpointing_enable()
        
        model.apply(selective_checkpoint)
        
        # Fusion des opérations
        if hasattr(torch.jit, '_script_if_tracing'):
            model = torch.jit.script(model)
        
        # Channels Last pour CNN (si applicable)
        if any(isinstance(m, nn.Conv2d) for m in model.modules()):
            model = model.to(memory_format=torch.channels_last)
        
        return model
    
    @staticmethod
    def get_dataloader_config() -> Dict[str, Any]:
        """Configuration optimale du DataLoader pour RTX 5090"""
        return {
            "num_workers": min(8, os.cpu_count() or 8),
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "multiprocessing_context": "spawn",
            # Pour dynamic batching
            "drop_last": False,
            "shuffle": True,
        }
    
    @staticmethod 
    def profile_training_step(model, batch_size: int = 32, seq_len: int = 2048):
        """
        Profile un step d'entraînement pour identifier les goulots d'étranglement
        
        Args:
            model: Le modèle à profiler
            batch_size: Taille du batch
            seq_len: Longueur de séquence
        """
        import torch.profiler as profiler
        
        # Créer des données factices
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dummy_input = torch.randint(0, 50000, (batch_size, seq_len)).to(device)
        
        # Configuration du profiler
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            schedule=profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=1
            ),
            on_trace_ready=profiler.tensorboard_trace_handler('./profiler_logs'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for _ in range(5):
                outputs = model(dummy_input)
                if isinstance(outputs, dict):
                    loss = outputs.get('loss', torch.tensor(0.0))
                elif isinstance(outputs, tuple):
                    loss = outputs[1] if len(outputs) > 1 else torch.tensor(0.0)
                else:
                    loss = torch.tensor(0.0)
                
                loss.backward()
                prof.step()
        
        print("📊 Profiling complete. View results with: tensorboard --logdir=./profiler_logs")
        
        # Afficher les statistiques
        print("\n🔍 Top 10 CUDA operations by time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        return prof


# Fonction helper pour configuration rapide
def setup_5090_training(args):
    """
    Configure automatiquement les paramètres pour RTX 5090
    
    Args:
        args: Arguments du script d'entraînement
    
    Returns:
        args modifiés avec optimisations RTX 5090
    """
    optimizer = RTX5090Optimizer()
    
    # Setup CUDA
    optimizer.setup_cuda_optimizations()
    
    # Get optimal config
    config = optimizer.get_optimal_config(
        model_type=getattr(args, 'model_type', 'mla'),
        size=getattr(args, 'size', 'large')
    )
    
    # Update args with optimal values
    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    
    print(f"✅ RTX 5090 optimizations applied:")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Block size: {args.block_size}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Precision: {args.dtype}")
    print(f"  - FP8 enabled: {args.use_fp8}")
    
    return args