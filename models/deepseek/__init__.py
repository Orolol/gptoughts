"""
Module DeepSeek pour l'entraînement de modèles LLM.
"""

# # Importer les classes nécessaires pour train.py
# try:
#     # D'abord essayer d'importer depuis deepseek_mini.py
#     from models.deepseek.deepseek_mini import DeepSeekMini, DeepSeekMiniConfig
# except ImportError:
#     # Sinon, utiliser l'adaptateur
from models.deepseek.deepseek_adapter import DeepSeekMini, DeepSeekMiniConfig

# Exporter les classes pour qu'elles soient accessibles via models.deepseek
__all__ = ['DeepSeekMini', 'DeepSeekMiniConfig']