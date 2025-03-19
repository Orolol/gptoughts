"""
Module d'optimisations pour l'entraînement des modèles LLM.
Ce module regroupe les différentes optimisations disponibles pour les GPUs.
"""

# Import des sous-modules
from optimization.fp8_optim import (
    check_fp8_support,
    is_fp8_compatible,
    ada_fp8_autocast,
    get_best_precision_for_ada,
    optimize_fp8_settings
)

from optimization.memory_optim import (
    preallocate_cuda_memory,
    cleanup_memory,
    print_memory_stats,
    set_memory_optim_env_vars,
    optimize_memory_for_device
)

from optimization.cuda_optim import (
    setup_cuda_optimizations,
    check_cuda_graph_compatibility,
    print_gpu_stats,
    set_seed,
    optimize_distributed_params,
    enable_flash_attention
)

from optimization.training_optim import (
    get_optimal_batch_size_for_gpu,
    AsyncDataLoader,
    optimize_attention_operations,
    optimize_training_parameters,
    enable_torch_compile,
    recover_from_nan,
    initialize_optimizer,
    autoconfigure_environment,
    patch_train_loop
)

__all__ = [
    # FP8 optimizations
    'check_fp8_support',
    'is_fp8_compatible',
    'ada_fp8_autocast',
    'get_best_precision_for_ada',
    'optimize_fp8_settings',
    
    # Memory optimizations
    'preallocate_cuda_memory',
    'cleanup_memory',
    'print_memory_stats',
    'set_memory_optim_env_vars',
    'optimize_memory_for_device',
    
    # CUDA optimizations
    'setup_cuda_optimizations',
    'check_cuda_graph_compatibility',
    'print_gpu_stats',
    'set_seed',
    'optimize_distributed_params',
    'enable_flash_attention',
    
    # Training optimizations
    'get_optimal_batch_size_for_gpu',
    'AsyncDataLoader',
    'optimize_attention_operations',
    'optimize_training_parameters',
    'enable_torch_compile',
    'recover_from_nan',
    'initialize_optimizer',
    'autoconfigure_environment',
    'patch_train_loop'
]

def configure_all_optimizations(use_fp8=False, prealloc_memory=True, enable_flash_attn=True, 
                               optimize_attention=True, optimize_batch=True, seed=42):
    """
    Configure toutes les optimisations en une seule fois
    
    Args:
        use_fp8: Activer les optimisations FP8 si disponible
        prealloc_memory: Préallouer la mémoire GPU pour éviter la fragmentation
        enable_flash_attn: Activer Flash Attention si disponible
        optimize_attention: Optimiser les opérations d'attention
        optimize_batch: Déterminer la taille de batch optimale
        seed: Graine pour la reproductibilité
    
    Returns:
        dict: Paramètres optimisés (taille de batch, etc.)
    """
    # Configurer l'environnement 
    autoconfigure_environment()
    
    # Optimisations CUDA de base
    setup_cuda_optimizations()
    
    # Optimisations mémoire
    optimize_memory_for_device()
    set_memory_optim_env_vars()
    
    if prealloc_memory:
        preallocate_cuda_memory()
    
    # Optimisations FP8
    if use_fp8:
        fp8_compatible = is_fp8_compatible()
        if fp8_compatible:
            optimize_fp8_settings()
            print("FP8 optimizations enabled")
        else:
            print("FP8 optimizations not enabled - hardware not compatible")
    
    # Attention optimizations
    if optimize_attention:
        if enable_flash_attn:
            enable_flash_attention()
        optimize_attention_operations()
    
    # Déterminisme
    set_seed(seed)
    
    # Optimisations distributed
    optimize_distributed_params()
    
    # Calculer les paramètres optimaux
    result = {}
    if optimize_batch:
        batch_size, grad_accum = get_optimal_batch_size_for_gpu()
        result['batch_size'] = batch_size
        result['gradient_accumulation_steps'] = grad_accum
    
    # Afficher les statistiques
    print_gpu_stats()
    print_memory_stats("Initial ")
    
    print("\nAll optimizations configured successfully!")
    return result 