#!/usr/bin/env python3
"""
Script de diagnostic pour vérifier l'installation de transformer-engine
et ses dépendances pour le support FP8.
"""

import os
import sys
import subprocess
from pathlib import Path
import platform
import importlib.util

def run_command(cmd):
    """Execute une commande et retourne le résultat"""
    try:
        result = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)
        return result.stdout.strip(), result.returncode
    except Exception as e:
        return str(e), 1

def check_package_installed(package_name):
    """Vérifie si un package est installé et retourne sa version"""
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            return False, None
        
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "Version inconnue")
        return True, version
    except ImportError:
        return False, None
    except Exception as e:
        return False, str(e)

def check_cuda_version():
    """Vérifie la version de CUDA installée"""
    try:
        import torch
        if not torch.cuda.is_available():
            return "CUDA non disponible dans PyTorch", False
        
        cuda_version = torch.version.cuda
        return f"CUDA version: {cuda_version}", True
    except ImportError:
        return "PyTorch non installé", False
    except Exception as e:
        return f"Erreur lors de la vérification de CUDA: {str(e)}", False

def check_transformer_engine_installation():
    """Vérifie l'installation de transformer-engine et ses dépendances"""
    print("=== Diagnostic de l'installation de transformer-engine ===")
    
    # Vérifier si transformer-engine est installé
    te_installed, te_version = check_package_installed("transformer_engine")
    if not te_installed:
        print("❌ transformer-engine n'est pas installé.")
        return False
    
    print(f"✓ transformer-engine est installé (version: {te_version})")
    
    # Vérifier l'installation de PyTorch
    torch_installed, torch_version = check_package_installed("torch")
    if not torch_installed:
        print("❌ PyTorch n'est pas installé, requis pour transformer-engine")
        return False
    
    print(f"✓ PyTorch est installé (version: {torch_version})")
    
    # Vérifier la version de CUDA
    cuda_msg, cuda_ok = check_cuda_version()
    print(f"{'✓' if cuda_ok else '❌'} {cuda_msg}")
    
    # Vérifier si les bibliothèques .so existent
    try:
        import transformer_engine.pytorch
        print("✓ Le module transformer_engine.pytorch peut être importé")
    except ImportError as e:
        print(f"❌ Erreur lors de l'import de transformer_engine.pytorch: {str(e)}")
        return False
    
    # Vérifier les bibliothèques .so
    try:
        import transformer_engine.pytorch as te
        module_path = Path(te.__file__).parent
        so_files = list(module_path.glob("*.so"))
        
        if not so_files:
            print(f"❌ Aucune bibliothèque .so trouvée dans {module_path}")
            return False
        
        print(f"✓ Bibliothèques .so trouvées dans {module_path}:")
        for so_file in so_files:
            print(f"  - {so_file.name}")
    except Exception as e:
        print(f"❌ Erreur lors de la vérification des bibliothèques .so: {str(e)}")
        return False
    
    # Essayer d'utiliser fp8_autocast
    try:
        hasattr(te, "fp8_autocast")
        print("✓ L'API fp8_autocast est disponible")
    except Exception as e:
        print(f"❌ L'API fp8_autocast n'est pas disponible: {str(e)}")
        return False
    
    return True

def suggest_fixes():
    """Suggère des corrections pour résoudre les problèmes d'installation"""
    print("\n=== Suggestions pour résoudre les problèmes ===")
    
    # Vérifier la compatibilité de la version CUDA
    cuda_msg, cuda_ok = check_cuda_version()
    if not cuda_ok:
        print("1. Assurez-vous que CUDA est correctement installé et accessible à PyTorch")
    
    # Vérifier l'architecture du processeur
    architecture = platform.machine()
    print(f"Architecture système détectée: {architecture}")
    
    # Vérifier la compatibilité de la version de PyTorch
    torch_installed, torch_version = check_package_installed("torch")
    if torch_installed:
        if "2.1" in torch_version or "2.2" in torch_version or "2.3" in torch_version or "2.4" in torch_version or "2.5" in torch_version:
            print("✓ Version de PyTorch compatible (2.1+)")
        else:
            print("⚠️ Version de PyTorch potentiellement incompatible. Version 2.1+ recommandée.")
    
    print("\nSuggestions de correction:")
    print("1. Réinstallez transformer-engine avec la bonne version de CUDA:")
    print("   pip uninstall -y transformer-engine")
    print("   pip install transformer-engine>=1.3.0")
    
    print("\n2. Si cela ne fonctionne pas, essayez d'installer depuis la source:")
    print("   git clone https://github.com/NVIDIA/TransformerEngine.git")
    print("   cd TransformerEngine")
    print("   python setup.py install")
    
    print("\n3. Vérifiez que votre GPU supporte FP8:")
    print("   Seuls les GPU NVIDIA H100, H200 ou avec architecture SM86+ (p. ex. RTX 3090 et +) supportent FP8")
    
    print("\n4. Si vous utilisez Docker, assurez-vous d'utiliser une image avec le bon support CUDA:")
    print("   Par exemple: nvcr.io/nvidia/pytorch:23.10-py3 ou plus récent")

if __name__ == "__main__":
    installation_ok = check_transformer_engine_installation()
    
    if not installation_ok:
        suggest_fixes()
        print("\nExécutez ce script après avoir appliqué les corrections pour vérifier l'installation.")
        sys.exit(1)
    else:
        print("\n✓ L'installation de transformer-engine semble correcte!")
        sys.exit(0) 