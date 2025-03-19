"""
Optimisations FP8 pour l'entraînement des modèles LLM.
Ce module contient les fonctions pour utiliser la précision FP8 sur les GPUs compatibles.
"""

import os
import sys
import torch
from contextlib import contextmanager

@contextmanager
def ada_fp8_autocast():
    """
    Contexte d'autocast adapté pour GPU Ada Lovelace (RTX 6000 Ada)
    avec fallback vers BF16 si FP8 n'est pas disponible
    """
    try:
        # Essayer d'importer transformer_engine pour FP8
        import transformer_engine.pytorch as te
        
        # Vérifier si fp8_autocast est disponible
        if not hasattr(te, "fp8_autocast"):
            # Le module existe mais pas l'API fp8_autocast
            print("Utilisation de BF16 - fp8_autocast non disponible dans transformer_engine")
            with torch.amp.autocast(enabled=True, device_type='cuda'):
                yield
        else:
            # Vérifier si recipe est disponible
            if hasattr(te, "common") and hasattr(te.common, "recipe"):
                # Configuration complète FP8 disponible
                from transformer_engine.common import recipe
                fp8_recipe = recipe.DelayedScaling(
                    margin=0,
                    interval=1,
                    fp8_format=recipe.Format.HYBRID
                )
                print("Utilisation de FP8 avec transformer_engine version récente")
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    yield
            else:
                # Version ancienne de transformer_engine (0.x)
                print("Utilisation de FP8 avec transformer_engine version ancienne")
                with te.fp8_autocast():
                    yield
    except ImportError:
        # transformer_engine non disponible
        print("Utilisation de BF16 - transformer_engine non disponible")
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            yield
    except Exception as e:
        # Autre erreur
        print(f"Utilisation de BF16 - erreur avec FP8: {e}")
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            yield
            
def get_best_precision_for_ada():
    """
    Détermine la meilleure précision disponible pour une RTX 6000 Ada
    
    Returns:
        str: 'fp8', 'bfloat16', ou 'float16'
    """
    # Vérifier si le GPU est compatible BF16
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    # Vérifier si FP8 est disponible
    fp8_supported = False
    try:
        import transformer_engine.pytorch
        if hasattr(transformer_engine.pytorch, "fp8_autocast"):
            # Vérifier si le GPU est Ada Lovelace
            if torch.cuda.get_device_properties(0).major >= 8:
                fp8_supported = True
    except ImportError:
        pass
    
    # Déterminer la meilleure précision
    if fp8_supported:
        return "fp8"
    elif bf16_supported:
        return "bfloat16"
    else:
        return "float16"
        
def check_fp8_support():
    """
    Vérifie le support FP8 de manière détaillée et imprime les résultats
    
    Returns:
        bool: True si FP8 est supporté, False sinon
    """
    print("\n=== Vérification du support FP8 pour RTX 6000 Ada ===")
    
    # Vérifier PyTorch
    try:
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"GPU détecté: {device_name}")
            
            # Vérifier l'architecture
            props = torch.cuda.get_device_properties(0)
            arch = f"SM {props.major}.{props.minor}"
            print(f"Architecture CUDA: {arch}")
            
            # Ada Lovelace est SM 8.9
            if props.major >= 8 and props.minor >= 9:
                print("✓ GPU Ada Lovelace détecté - support FP8 partiel possible")
            elif props.major >= 9:
                print("✓ GPU Hopper ou plus récent détecté - support FP8 complet")
            else:
                print("⚠️ GPU plus ancien détecté - FP8 non supporté nativement")
                return False
        else:
            print("❌ CUDA non disponible")
            return False
    except Exception as e:
        print(f"❌ Erreur lors de la vérification de PyTorch: {e}")
        return False
    
    # Vérifier transformer-engine
    try:
        import transformer_engine
        print(f"transformer-engine version: {getattr(transformer_engine, '__version__', 'inconnue')}")
        
        # Vérifier les sous-modules
        if hasattr(transformer_engine, 'pytorch'):
            print("✓ Module transformer_engine.pytorch disponible")
            
            # Vérifier fp8_autocast
            if hasattr(transformer_engine.pytorch, 'fp8_autocast'):
                print("✓ API fp8_autocast disponible")
                return True
            else:
                print("❌ API fp8_autocast non disponible")
                return False
        else:
            print("❌ Module transformer_engine.pytorch non disponible")
            return False
    except ImportError:
        print("❌ transformer-engine non installé")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de la vérification de transformer-engine: {e}")
        return False

def is_fp8_compatible():
    """Vérifie si le GPU est compatible avec FP8 (H100, H200)"""
    if not torch.cuda.is_available():
        return False
    try:
        # FP8 est supporté sur Hopper (SM90) et certains Ampere supérieurs (SM86+)
        props = torch.cuda.get_device_properties(0)
        return (props.major >= 9) or (props.major == 8 and props.minor >= 6)
    except:
        return False

def optimize_fp8_settings(args=None):
    """
    Optimise les paramètres pour l'entraînement en FP8
    
    Args:
        args: Arguments de l'entraînement (optionnel)
    
    Returns:
        Arguments mis à jour si fournis, sinon None
    """
    # Vérifier la compatibilité FP8
    is_compatible = is_fp8_compatible()
    
    # Configurer les variables d'environnement pour FP8
    if is_compatible:
        os.environ["NVTE_ALLOW_FP8_USAGE"] = "1"
        os.environ["NVTE_FP8_RECIPE_FORMAT"] = "HYBRID"  # ou E4M3
        os.environ["NVTE_FP8_MHA"] = "1"
        os.environ["NVTE_FP8_FFN"] = "1"
        print("✓ Paramètres d'environnement FP8 optimisés activés")
    
    # Si des arguments sont fournis, les configurer pour FP8
    if args is not None and is_compatible:
        if hasattr(args, "use_fp8") and not args.use_fp8:
            print("! Option use_fp8 explicitement désactivée dans les arguments")
        else:
            if hasattr(args, "use_fp8"):
                args.use_fp8 = True
            
            if hasattr(args, "precision"):
                args.precision = "fp8"
    
    return args if args is not None else None

if __name__ == "__main__":
    # Vérifier le support FP8
    fp8_supported = check_fp8_support()
    
    # Afficher le résultat
    if fp8_supported:
        print("\n✅ FP8 est supporté sur ce système!")
        print("Vous pouvez utiliser --use_fp8 pour activer l'entraînement en FP8")
    else:
        print("\n⚠️ FP8 n'est pas entièrement supporté sur ce système.")
        print("L'entraînement utilisera automatiquement BF16 comme fallback")
        print("Cela reste une très bonne précision pour l'entraînement!") 