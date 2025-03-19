"""
Optimisations spécifiques à l'entraînement des modèles LLM.
Ce module regroupe les fonctionnalités avancées pour l'optimisation de l'entraînement.
"""

import os
import sys
import torch
import argparse
import gc
from contextlib import nullcontext

def get_optimal_batch_size_for_gpu(model_type="llada", size="medium", min_batch=8, max_batch=64):
    """
    Détermine automatiquement la taille de batch optimale en fonction de la mémoire GPU disponible
    
    Args:
        model_type: Type de modèle
        size: Taille du modèle
        min_batch: Taille de batch minimale à considérer
        max_batch: Taille de batch maximale à considérer
        
    Returns:
        tuple: (batch_size, gradient_accumulation_steps)
    """
    if not torch.cuda.is_available():
        print("GPU non disponible, utilisation des valeurs par défaut")
        return 8, 4
    
    # Obtenir la mémoire disponible
    try:
        free_mem, total_mem = torch.cuda.mem_get_info()
        mem_gb = total_mem / (1024**3)
    except:
        # Fallback pour les versions plus anciennes de PyTorch
        props = torch.cuda.get_device_properties(0)
        mem_gb = props.total_memory / (1024**3)
    
    print(f"Mémoire GPU détectée: {mem_gb:.1f} GB")
    
    # Estimer les besoins en mémoire selon le modèle et sa taille
    mem_multipliers = {
        "gpt": {"small": 1.0, "medium": 1.5, "large": 2.5},
        "llada": {"small": 1.2, "medium": 1.8, "large": 3.0},
        "deepseek": {"small": 1.1, "medium": 1.7, "large": 2.8}
    }
    
    # Valeurs par défaut si le modèle ou la taille n'est pas reconnue
    model_type = model_type.lower() if model_type in mem_multipliers else "llada"
    size = size.lower() if size in mem_multipliers[model_type] else "medium"
    
    # Facteur de mémoire pour le modèle spécifié
    mem_factor = mem_multipliers[model_type][size]
    
    # Calculer la taille de batch optimale
    # Formule empirique basée sur des tests
    if mem_gb < 12:
        # GPUs avec moins de 12GB (RTX 3060, etc.)
        batch_size = max(min_batch, int(min(max_batch, (mem_gb - 2) / mem_factor)))
        grad_accum = max(1, int(24 / batch_size))
    elif mem_gb < 24:
        # GPUs milieu de gamme (RTX 3090, 4090, A5000, etc.)
        batch_size = max(min_batch, int(min(max_batch, (mem_gb - 4) / mem_factor)))
        grad_accum = max(1, int(32 / batch_size))
    elif mem_gb < 40:
        # GPUs haut de gamme (A6000, etc.)
        batch_size = max(min_batch, int(min(max_batch, (mem_gb - 6) / mem_factor)))
        grad_accum = max(1, int(48 / batch_size))
    else:
        # GPUs de datacenter (A100, H100, etc.)
        batch_size = max(min_batch, int(min(max_batch, (mem_gb - 10) / mem_factor)))
        grad_accum = max(1, int(64 / batch_size))
    
    # Ajuster pour les modèles MoE qui consomment plus de mémoire
    if model_type == "llada" and mem_gb < 32:
        batch_size = max(min_batch, batch_size - 4)
    
    print(f"Taille de batch optimale calculée: {batch_size}, accum. gradient: {grad_accum}")
    return batch_size, grad_accum

class AsyncDataLoader:
    """
    Wrapper pour charger les données de manière asynchrone sur GPU,
    en préchargeant les prochains batches en parallèle.
    """
    def __init__(self, dataset, buffer_size=2):
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.buffer = []
        self.current_idx = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._fill_buffer()
    
    def _fill_buffer(self):
        """Remplit le buffer avec des données préchargées"""
        while len(self.buffer) < self.buffer_size:
            try:
                data = next(self.dataset)
                # Mettre les données sur GPU directement
                if isinstance(data, torch.Tensor):
                    self.buffer.append(data.to(self.device, non_blocking=True))
                elif isinstance(data, tuple) and all(isinstance(t, torch.Tensor) for t in data):
                    self.buffer.append(tuple(t.to(self.device, non_blocking=True) for t in data))
                else:
                    self.buffer.append(data)  # Fallback pour les données non-tensors
            except StopIteration:
                break
    
    def __iter__(self):
        self.current_idx = 0
        self.buffer = []
        self._fill_buffer()
        return self
    
    def __next__(self):
        if not self.buffer:
            raise StopIteration
        
        # Récupérer le prochain élément du buffer
        item = self.buffer.pop(0)
        
        # Précharger le prochain élément en parallèle
        try:
            next_data = next(self.dataset)
            if isinstance(next_data, torch.Tensor):
                self.buffer.append(next_data.to(self.device, non_blocking=True))
            elif isinstance(next_data, tuple) and all(isinstance(t, torch.Tensor) for t in next_data):
                self.buffer.append(tuple(t.to(self.device, non_blocking=True) for t in next_data))
            else:
                self.buffer.append(next_data)
        except StopIteration:
            pass
        
        return item

def optimize_attention_operations():
    """
    Optimise les opérations d'attention pour un meilleur débit.
    Active Flash Attention, optimise le scaled dot-product attention, etc.
    """
    # Vérifier si transformers est disponible
    try:
        import transformers
        
        # Activer Flash Attention si disponible
        if hasattr(transformers, "utils") and hasattr(transformers.utils, "is_flash_attn_available"):
            if transformers.utils.is_flash_attn_available():
                os.environ["TRANSFORMERS_ENABLE_FLASH_ATTN"] = "1"
                print("Flash Attention activé via transformers")
        
        # Utiliser la version la plus rapide des opérations d'attention
        if hasattr(transformers, "utils") and hasattr(transformers.utils, "is_torch_sdpa_available"):
            if transformers.utils.is_torch_sdpa_available():
                os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "sdpa"
                print("Utilisation de SDPA (Scaled Dot-Product Attention) optimisé")
    except ImportError:
        print("Transformers non disponible, optimisations d'attention limitées")
    
    # Optimisations via variables d'environnement
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"  # Activer l'API CUDNNv8 si disponible
    
    # Optimisations spécifiques aux opérations d'attention dans PyTorch 2.0+
    if hasattr(torch, "__version__") and int(torch.__version__.split('.')[0]) >= 2:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        print("Optimisations SDP (Scaled Dot-Product) PyTorch 2.0+ activées")

def optimize_training_parameters(args):
    """
    Optimise les paramètres d'entraînement en fonction du modèle et du matériel
    
    Args:
        args: Arguments d'entraînement (namespace d'argparse)
        
    Returns:
        args: Arguments mis à jour
    """
    # Si args n'a pas les attributs ou s'ils ne sont pas spécifiés, les calculer automatiquement
    if not hasattr(args, 'batch_size') or args.batch_size <= 0 or getattr(args, 'batch_size_auto_optimize', False):
        bs, ga = get_optimal_batch_size_for_gpu(
            model_type=getattr(args, 'model_type', 'llada'),
            size=getattr(args, 'size', 'medium')
        )
        args.batch_size = bs
        print(f"Taille de batch optimisée automatiquement: {bs}")
        
        # Ne pas écraser grad_accum si déjà défini par l'utilisateur
        if not hasattr(args, 'gradient_accumulation_steps') or args.gradient_accumulation_steps <= 0:
            args.gradient_accumulation_steps = ga
            print(f"Accumulation de gradient optimisée automatiquement: {ga}")
    
    # Optimisations spécifiques au modèle
    if hasattr(args, 'model_type'):
        if args.model_type.lower() == 'llada':
            # LLaDA nécessite un coefficient de perte de routeur plus faible pour la stabilité
            if not hasattr(args, 'router_z_loss_coef') or args.router_z_loss_coef > 0.0005:
                args.router_z_loss_coef = 0.0001
                print(f"Coefficient de perte du routeur pour LLaDA optimisé: {args.router_z_loss_coef}")
            
            # LLaDA fonctionne mieux avec Apollo comme optimiseur
            if not hasattr(args, 'optimizer_type') or not args.optimizer_type:
                args.optimizer_type = 'apollo-mini'
                print(f"Optimiseur pour LLaDA configuré: {args.optimizer_type}")
        
        elif args.model_type.lower() == 'deepseek':
            # DeepSeek nécessite plus de warmup
            if not hasattr(args, 'warmup_iters') or args.warmup_iters < 100:
                args.warmup_iters = 100
                print(f"Warmup pour DeepSeek optimisé: {args.warmup_iters}")
    
    # Optimisations pour éviter les NaN
    if not hasattr(args, 'grad_clip') or args.grad_clip == 0.0:
        args.grad_clip = 1.0
        print(f"Clip de gradient configuré pour stabilité: {args.grad_clip}")
    
    if not hasattr(args, 'compile'):
        args.compile = hasattr(torch, 'compile')
        if args.compile:
            print("Compilation PyTorch activée pour de meilleures performances")
    
    # Sélection du type de données optimal
    if not hasattr(args, 'dtype') or not args.dtype:
        if torch.cuda.is_bf16_supported():
            args.dtype = 'bfloat16'
            print("BFloat16 détecté et activé (meilleure stabilité numérique)")
        else:
            args.dtype = 'float16'
            print("Float16 activé (GPU ne supportant pas BFloat16)")
    
    return args

def enable_torch_compile(model, precision='default', mode='max-autotune', backend=None):
    """
    Active torch.compile pour accélérer l'exécution du modèle.
    
    Args:
        model: Le modèle à compiler
        precision: Précision à utiliser ('default', 'highest', 'reduced')
        mode: Mode de compilation ('default', 'reduce-overhead', 'max-autotune')
        backend: Backend spécifique ('inductor', 'aot_eager', etc.)
        
    Returns:
        model: Le modèle compilé
    """
    if not torch.cuda.is_available():
        print("CUDA non disponible, compilation ignorée")
        return model
        
    if not hasattr(torch, 'compile'):
        print("torch.compile non disponible (nécessite PyTorch 2.0+)")
        return model
    
    # Configurer le backend
    if backend is None:
        backend = "inductor"  # Meilleur backend par défaut dans PyTorch 2.0+
    
    # Configurer les options de compilation
    compile_options = {}
    
    # Différents modes d'optimisation
    if mode == 'reduce-overhead':
        compile_options['fullgraph'] = False
        compile_options['dynamic'] = True
    elif mode == 'max-autotune':
        compile_options['fullgraph'] = True
        compile_options['dynamic'] = False
    
    # Configurer la précision
    if precision == 'highest':
        compile_options['mode'] = 'max-precision'
    elif precision == 'reduced':
        compile_options['mode'] = 'reduce-precision'
    
    print(f"Compilation du modèle avec backend '{backend}' et mode '{mode}'")
    
    try:
        compiled_model = torch.compile(model, backend=backend, **compile_options)
        print("Modèle compilé avec succès!")
        return compiled_model
    except Exception as e:
        print(f"Erreur lors de la compilation: {e}")
        print("Utilisation du modèle non compilé")
        return model

def recover_from_nan(model, optimizer, training_args, last_loss):
    """
    Tente de récupérer d'une perte NaN pendant l'entraînement
    
    Args:
        model: Le modèle en cours d'entraînement
        optimizer: L'optimiseur
        training_args: Arguments d'entraînement
        last_loss: Dernière perte valide
        
    Returns:
        bool: True si la récupération a réussi, False sinon
    """
    print("\n===== DÉTECTION DE PERTE NaN - TENTATIVE DE RÉCUPÉRATION =====")
    
    # 1. Nettoyer la mémoire
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # 2. Réduire le learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5
    reduced_lr = optimizer.param_groups[0]['lr']
    print(f"Learning rate réduit à {reduced_lr}")
    
    # 3. Augmenter le clipping de gradient
    if hasattr(training_args, 'grad_clip'):
        old_clip = training_args.grad_clip
        training_args.grad_clip = min(old_clip * 2.0, 5.0)
        print(f"Clipping de gradient augmenté de {old_clip} à {training_args.grad_clip}")
    else:
        training_args.grad_clip = 1.0
        print(f"Clipping de gradient activé: {training_args.grad_clip}")
    
    # 4. Vérifier les paramètres du modèle pour les NaN
    has_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"Paramètre {name} contient des NaN!")
            has_nan = True
            # Essayer de remplir les NaN avec des zéros
            nan_mask = torch.isnan(param)
            param.data[nan_mask] = 0.0
    
    if has_nan:
        print("NaN détectés et remplacés par des zéros dans les paramètres du modèle")
    
    # 5. Si le modèle est un MoE (comme LLaDA), réduire le coefficient de perte du routeur
    if hasattr(training_args, 'model_type') and training_args.model_type.lower() == 'llada':
        if hasattr(training_args, 'router_z_loss_coef'):
            old_coef = training_args.router_z_loss_coef
            training_args.router_z_loss_coef *= 0.1
            print(f"Coefficient de perte du routeur réduit de {old_coef} à {training_args.router_z_loss_coef}")
    
    print("Paramètres d'entraînement ajustés pour la stabilité")
    print("===== FIN DE LA PROCÉDURE DE RÉCUPÉRATION =====\n")
    
    return not has_nan  # Retourne True si aucun NaN n'a été trouvé dans les paramètres

def initialize_optimizer(model, args):
    """
    Initialise l'optimiseur optimal en fonction du type de modèle
    
    Args:
        model: Le modèle à optimiser
        args: Arguments de configuration
        
    Returns:
        optimizer: L'optimiseur configuré
    """
    # Paramètres d'optimisation
    lr = getattr(args, 'learning_rate', 3e-5)
    weight_decay = getattr(args, 'weight_decay', 0.01)
    beta1 = getattr(args, 'beta1', 0.9)
    beta2 = getattr(args, 'beta2', 0.95)
    
    # Déterminer le type d'optimiseur par défaut selon le modèle
    if not hasattr(args, 'optimizer_type') or args.optimizer_type is None:
        if hasattr(args, 'model_type'):
            if args.model_type.lower() in ['llada', 'deepseek']:
                args.optimizer_type = 'apollo-mini'
            else:
                args.optimizer_type = 'adamw'
        else:
            args.optimizer_type = 'adamw'
    
    print(f"Initialisation de l'optimiseur: {args.optimizer_type}")
    
    # Paramètres pour l'optimiseur
    params = model.parameters()
    
    # Sélectionner l'optimiseur
    if args.optimizer_type == 'lion':
        try:
            from lion_pytorch import Lion
            optimizer = Lion(params, lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
        except ImportError:
            print("Lion non disponible, retour à AdamW")
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
    
    elif args.optimizer_type in ['apollo', 'apollo-mini']:
        try:
            sys.path.append('.')
            if args.optimizer_type == 'apollo-mini':
                from optimizers.apollo import ApolloMini as Apollo
            else:
                from optimizers.apollo import Apollo
            
            optimizer = Apollo(params, lr=lr, weight_decay=weight_decay, beta=beta1)
        except ImportError:
            print("Apollo non disponible, retour à AdamW")
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
    
    else:  # 'adamw' ou tout autre valeur
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
    
    return optimizer

def autoconfigure_environment():
    """
    Configure automatiquement les variables d'environnement pour un entraînement optimal
    """
    # Variables d'environnement générales
    os.environ["OMP_NUM_THREADS"] = str(max(1, os.cpu_count() // 2))
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    # Optimisations PyTorch
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    
    # Configuration de l'attention
    os.environ["TRANSFORMERS_ENABLE_FLASH_ATTN"] = "1"
    os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "sdpa"
    
    # Détection des GPUs 
    if torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).major >= 8:
            # GPUs récents (Ampere+)
            os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
            print("Détection de GPU Ampere+ (SM80+)")
        else:
            os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
        
        # Vérifier la capacité de traitement des tenseurs
        if torch.cuda.get_device_properties(0).major >= 7:
            # GPUs avec Tensor Cores
            os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
            print("Tensor Cores détectés, TF32 activé")

def patch_train_loop(trainer_class):
    """
    Applique un patch à la boucle d'entraînement du Trainer pour optimiser les performances
    
    Args:
        trainer_class: La classe Trainer à modifier
    """
    # Sauvegarder la méthode originale
    original_train = trainer_class.train
    
    def optimized_train(self):
        """Version optimisée de la méthode train"""
        # Appliquer les optimisations spécifiques à l'entraînement
        if torch.cuda.is_available():
            print("Optimisation de la boucle d'entraînement...")
            
            # Synchroniser avant de commencer
            torch.cuda.synchronize()
            
            # Déplacer tous les poids du modèle sur GPU si nécessaire
            for param in self.model.parameters():
                if param.device.type != 'cuda':
                    param.data = param.data.to(getattr(self, 'device', 'cuda'))
            
            # Créer des CUDA streams pour paralléliser les opérations
            default_stream = torch.cuda.current_stream()
            compute_stream = torch.cuda.Stream()
            
            # Stocker les streams dans l'instance pour y accéder plus tard
            self.compute_stream = compute_stream
            self.default_stream = default_stream
            
            # Activer le profiler CUDA si disponible
            if hasattr(torch.cuda, 'cudart'):
                torch.cuda.cudart().cudaProfilerStart()
        
        # Appeler la méthode originale pour l'entraînement
        result = original_train(self)
        
        # Nettoyage après l'entraînement
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            # Désactiver le profiler CUDA si activé
            if hasattr(torch.cuda, 'cudart'):
                torch.cuda.cudart().cudaProfilerStop()
        
        return result
    
    # Remplacer la méthode originale par la version optimisée
    trainer_class.train = optimized_train
    print("Boucle d'entraînement optimisée appliquée")
    
    return trainer_class

if __name__ == "__main__":
    print("Outils d'optimisation d'entraînement disponibles:")
    print("- get_optimal_batch_size_for_gpu(): Détermine la taille de batch optimale")
    print("- AsyncDataLoader: Charge les données de manière asynchrone")
    print("- optimize_attention_operations(): Optimise les opérations d'attention")
    print("- optimize_training_parameters(): Optimise les paramètres d'entraînement")
    print("- enable_torch_compile(): Compile le modèle pour de meilleures performances")
    print("- recover_from_nan(): Récupère d'une perte NaN")
    print("- initialize_optimizer(): Initialise l'optimiseur optimal")
    print("- autoconfigure_environment(): Configure l'environnement automatiquement")
    print("- patch_train_loop(): Optimise la boucle d'entraînement") 