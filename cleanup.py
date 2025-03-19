#!/usr/bin/env python3
"""
Script pour déplacer les fichiers obsolètes vers le dossier legacy_scripts
"""

import os
import shutil
from datetime import datetime

# Dossier de destination
LEGACY_DIR = "legacy_scripts"

# Créer le dossier s'il n'existe pas
os.makedirs(LEGACY_DIR, exist_ok=True)

# Créer un fichier README dans le dossier legacy_scripts
with open(os.path.join(LEGACY_DIR, "README.md"), "w") as f:
    f.write("""# Scripts obsolètes

Ce dossier contient des scripts et fichiers qui ont été migrés vers la nouvelle structure d'optimisation.  
Ces fichiers sont conservés pour référence mais ne devraient plus être utilisés.

Voir `../OPTIMIZATION_README.md` pour plus d'informations sur la nouvelle structure.

Date d'archivage : {}
""".format(datetime.now().strftime("%Y-%m-%d")))

# Liste des fichiers Python obsolètes
python_files = [
    "ada_fp8_utils.py", 
    "gpu_optimization.py",
    "gpu_optimization_advanced.py",
    "gpu_optimization_enhanced.py",
    "run_train_enhanced.py",
    "check_transformer_engine.py",
]

# Liste des scripts shell obsolètes
shell_files = [
    "install_ada_fp8.sh",
    "install_fp8_support.sh",
    "install_transformer_engine_complete.sh",
    "reinstall_transformer_engine.sh",
    "recover_from_nan.sh",
    "recover_from_nan_enhanced.sh",
    "train_deepseek_optimized.sh",
    "train_deepseek_stable.sh",
    "train_fp8.sh", 
    "train_max_gpu.sh",
    "train_optimized.sh",
    "install_te_official.sh",
]

# Liste des fichiers markdown obsolètes
markdown_files = [
    "ADVANCED_GPU_OPTIMIZATION.md",
    "GPU_OPTIMIZATION_README.md",
    "recommended_fix.md",
]

# Liste des fichiers de requirements obsolètes
requirements_files = [
    "requirements-fp8-ada.txt",
    "requirements-fp8.txt",
    "requirements-turing.txt",
]

# Combiner toutes les listes
all_files = python_files + shell_files + markdown_files + requirements_files

# Déplacer les fichiers
moved_files = []
for file in all_files:
    src_path = file
    dst_path = os.path.join(LEGACY_DIR, file)
    
    # Vérifier si le fichier existe
    if os.path.exists(src_path):
        try:
            shutil.move(src_path, dst_path)
            moved_files.append(file)
            print(f"✓ Déplacé: {file}")
        except Exception as e:
            print(f"❌ Erreur lors du déplacement de {file}: {e}")
    else:
        print(f"⚠️ Fichier non trouvé: {file}")

print(f"\n{len(moved_files)} fichiers ont été déplacés vers {LEGACY_DIR}/")
print("Un fichier README.md a été ajouté dans ce dossier pour expliquer le contexte.")
print("\nRappelez-vous : les fonctionnalités de ces scripts sont maintenant disponibles via:")
print("- ./optimize.sh --help  (pour voir toutes les options)")
print("- import optimization  (dans vos scripts Python)") 