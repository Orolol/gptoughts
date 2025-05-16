"""
Module DeepSeek pour l'entraînement de modèles LLM.
"""

# Importer les classes de l'adaptateur original (inference only)
from models.deepseek.deepseek_adapter import DeepSeekMini as DeepSeekMiniInference
from models.deepseek.deepseek_adapter import DeepSeekMiniConfig

# Importer les classes de l'adaptateur trainable
from models.deepseek.deepseek_adapter_trainable import DeepSeekMini as DeepSeekMiniTrainable

# Importer les classes de l'adaptateur trainable avec MTP
from models.deepseek.deepseek_adapter_mtp import DeepSeekMiniMTP, DeepSeekMiniConfigMTP

# Par défaut, utiliser la version trainable avec MTP (compatible avec inference et training)
DeepSeekMini = DeepSeekMiniMTP

# Exporter les classes pour qu'elles soient accessibles via models.deepseek
__all__ = [
    'DeepSeekMini',
    'DeepSeekMiniConfig',
    'DeepSeekMiniConfigMTP',
    'DeepSeekMiniInference',
    'DeepSeekMiniTrainable',
    'DeepSeekMiniMTP'
]