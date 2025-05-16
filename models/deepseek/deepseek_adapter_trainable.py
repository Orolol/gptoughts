"""
Adaptateur amélioré pour le modèle DeepSeek.
Ce fichier permet d'utiliser le modèle DeepSeek avec support d'entraînement.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple, Union

from models.deepseek.deepseek import ModelArgs
from models.deepseek.deepseek_trainable import DeepSeekTrainable
from train.train_utils import estimate_mfu as utils_estimate_mfu

class DeepSeekMiniConfig:
    """
    Configuration pour le modèle DeepSeek adaptée au format attendu par train.py.
    """
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        head_dim: int = 128,
        intermediate_size: int = 4096,
        num_experts: int = 32,
        num_experts_per_token: int = 4,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        layernorm_epsilon: float = 1e-6,
        kv_compression_dim: int = 128,
        query_compression_dim: int = 384,
        rope_head_dim: int = 32,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        bias: bool = False,
        expert_bias_init: float = 0.0,
        expert_bias_update_speed: float = 0.001,
        expert_balance_factor: float = 0.0001,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = True,
        use_return_dict: bool = True,
        router_z_loss_coef: float = 0.001,
    ):
        # Paramètres pour ModelArgs
        self.max_batch_size = 8
        self.max_seq_len = max_position_embeddings
        self.dtype = "bf16"
        self.vocab_size = vocab_size
        self.dim = hidden_size
        self.inter_dim = intermediate_size
        self.moe_inter_dim = intermediate_size // 8  # Approximation
        self.n_layers = num_hidden_layers
        self.n_dense_layers = 1  # Par défaut
        self.n_heads = num_attention_heads
        
        # Paramètres MoE
        self.n_routed_experts = num_experts
        self.n_shared_experts = 2  # Par défaut
        self.n_activated_experts = num_experts_per_token
        self.n_expert_groups = 1
        self.n_limited_groups = 1
        self.score_func = "softmax"
        self.route_scale = 1.0
        
        # Paramètres MLA
        self.q_lora_rank = 0
        self.kv_lora_rank = 512
        self.qk_nope_head_dim = 128
        self.qk_rope_head_dim = rope_head_dim
        self.v_head_dim = 128
        
        # Paramètres YARN
        self.original_seq_len = 4096
        self.rope_theta = rope_theta
        self.rope_factor = 40
        self.beta_fast = 32
        self.beta_slow = 1
        self.mscale = 1.0
        
        # Paramètres supplémentaires pour train.py
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.bias = bias
        self.layernorm_epsilon = layernorm_epsilon
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_cache = use_cache
        self.use_return_dict = use_return_dict
        self.router_z_loss_coef = router_z_loss_coef

class DeepSeekMini(nn.Module):
    """
    Adaptateur pour le modèle DeepSeek avec support d'entraînement.
    Cette classe enveloppe le modèle DeepSeekTrainable pour le rendre compatible
    avec l'interface attendue par train.py.
    """
    def __init__(self, config: DeepSeekMiniConfig):
        super().__init__()
        # Convertir la config DeepSeekMiniConfig en ModelArgs
        model_args = ModelArgs(
            max_batch_size=config.max_batch_size,
            max_seq_len=config.max_seq_len,
            dtype=config.dtype,
            vocab_size=config.vocab_size,
            dim=config.dim,
            inter_dim=config.inter_dim,
            moe_inter_dim=config.moe_inter_dim,
            n_layers=config.n_layers,
            n_dense_layers=config.n_dense_layers,
            n_heads=config.n_heads,
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=config.n_shared_experts,
            n_activated_experts=config.n_activated_experts,
            n_expert_groups=config.n_expert_groups,
            n_limited_groups=config.n_limited_groups,
            score_func=config.score_func,
            route_scale=config.route_scale,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            original_seq_len=config.original_seq_len,
            rope_theta=config.rope_theta,
            rope_factor=config.rope_factor,
            beta_fast=config.beta_fast,
            beta_slow=config.beta_slow,
            mscale=config.mscale,
        )
        
        # Créer le modèle DeepSeekTrainable (avec support d'entraînement)
        self.model = DeepSeekTrainable(model_args)
        self.config = config
        
        # Stocker la configuration pour référence
        self.model_args = model_args
        
        # Variables pour le timing et le gradient checkpointing
        self.timing_stats = None
        self.gradient_checkpointing = False
    
    def get_input_embeddings(self) -> nn.Module:
        """Retourne la couche d'embedding du modèle."""
        return self.model.transformer.embed
    
    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Définit la couche d'embedding du modèle."""
        self.model.transformer.embed = value
    
    def set_gradient_checkpointing(self, value: bool):
        """Active ou désactive le gradient checkpointing."""
        self.gradient_checkpointing = value
        self.model.set_gradient_checkpointing(value)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Dict[str, torch.Tensor]]:
        """
        Méthode forward adaptée pour être compatible avec l'interface attendue par train.py.
        
        Args:
            input_ids: Tensor d'indices de tokens en entrée
            attention_mask: Masque d'attention (ignoré dans cette implémentation)
            position_ids: Indices de position (ignorés)
            past_key_values: Cache KV pour la génération (ignoré)
            inputs_embeds: Embeddings pré-calculés (ignoré)
            use_cache: Activer la mise en cache (ignoré)
            output_attentions: Retourner les scores d'attention (ignoré)
            output_hidden_states: Retourner les états cachés (ignoré)
            return_dict: Retourner un dictionnaire (sinon un tuple)
            targets: Tensor des indices de tokens cibles pour le calcul de perte
        
        Returns:
            Logits, perte et perte d'équilibrage (pour les modèles MoE)
        """
        # Ignorer les paramètres non utilisés
        _ = attention_mask, position_ids, past_key_values
        _ = inputs_embeds, use_cache, output_attentions, output_hidden_states
        
        # Vérifier les dimensions d'entrée pour éviter les problèmes numériques
        if input_ids.size(1) < 2:  # Si la séquence est trop courte
            # Padding pour assurer une longueur minimale de 2 (pour le décalage)
            pad_length = 2 - input_ids.size(1)
            padding = torch.zeros(input_ids.size(0), pad_length, dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([input_ids, padding], dim=1)
            if targets is not None:
                targets = torch.cat([targets, padding], dim=1)
        
        # S'assurer que le modèle est en mode approprié
        self.model.train(self.training)
        
        # Appeler le modèle trainable en mode approprié
        try:
            # Forward pass du modèle avec gestion des exceptions
            if targets is not None:
                # Mode entraînement avec calcul de perte
                output = self.model(input_ids, targets=targets, start_pos=0)
                logits, loss = output  # Déstructurer le tuple (logits, loss)
                
                # Balance loss (pertes auxiliaires pour MoE)
                # Le modèle DeepSeek n'utilise pas de balance_loss directement,
                # mais nous fournissons une valeur factice pour compatibilité avec train.py
                balance_loss = torch.tensor(0.0, device=loss.device)
                
            else:
                # Mode inférence, seulement les logits
                logits = self.model(input_ids, start_pos=0)
                loss = None
                balance_loss = None
                
        except RuntimeError as e:
            # En cas d'erreur, afficher un message et retourner une perte élevée
            print(f"Forward pass error: {e}")
            if return_dict:
                return {
                    "last_hidden_state": None,
                    "loss": torch.tensor(100.0, device=input_ids.device),
                    "balance_loss": torch.tensor(0.0, device=input_ids.device)
                }
            else:
                return None, torch.tensor(100.0, device=input_ids.device)
        
        # Retourner le résultat dans le format demandé
        if return_dict:
            return {
                "last_hidden_state": logits,
                "loss": loss,
                "balance_loss": balance_loss if balance_loss is not None else torch.tensor(0.0, device=logits.device)
            }
        else:
            return (logits, loss)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        **kwargs
    ) -> torch.LongTensor:
        """
        Génère du texte à partir des tokens d'entrée.
        
        Args:
            input_ids: Tensor des indices de tokens en entrée
            max_new_tokens: Nombre maximum de nouveaux tokens à générer
            temperature: Facteur de température pour l'échantillonnage
            top_k: Nombre de tokens à considérer pour l'échantillonnage top-k
        
        Returns:
            Tensor des tokens générés
        """
        # Mettre le modèle en mode évaluation
        self.model.eval()
        
        # Appeler la méthode de génération du modèle
        with torch.no_grad():
            # Obtenir les logits pour le premier token
            logits = self.model(input_ids, start_pos=0)
            
            # Générer des tokens de manière auto-régressive
            for _ in range(max_new_tokens):
                # Appliquer la température et le top-k si nécessaire
                if temperature > 0:
                    probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                    if top_k is not None and top_k > 0:
                        v, _ = torch.topk(probs, top_k)
                        probs[probs < v[:, [-1]]] = 0
                        probs.div_(probs.sum(dim=-1, keepdim=True))
                    # Échantillonner le prochain token
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Génération déterministe (greedy decoding)
                    next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
                
                # Ajouter le token généré à l'entrée
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Générer les logits pour le prochain token
                logits = self.model(input_ids, start_pos=0)
            
            return input_ids
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, **kwargs):
        """
        Configure l'optimiseur pour l'entraînement.
        
        Args:
            weight_decay: Facteur de décroissance des poids
            learning_rate: Taux d'apprentissage
            betas: Paramètres betas pour Adam
            device_type: Type de device ('cuda' ou 'cpu')
            
        Returns:
            Optimiseur configuré
        """
        # Déléguer la configuration de l'optimiseur au modèle sous-jacent
        return self.model.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
            device_type=device_type,
            **kwargs
        )
    
    def set_timing_stats(self, timing_stats):
        """Définit les statistiques de timing pour le profilage."""
        self.timing_stats = timing_stats
    
    def estimate_mfu(self, batch_size: int, dt: float) -> float:
        """Estime l'utilisation des FLOPS du modèle (MFU) en pourcentage."""
        # Déléguer l'estimation au modèle sous-jacent
        return self.model.estimate_mfu(
            batch_size=batch_size,
            seq_length=self.model_args.max_seq_len,
            dt=dt
        )