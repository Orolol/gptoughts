"""
Adaptive Attention MoE Components
MoE applied to attention mechanism with importance-based routing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class LanguageFeatureRouter(nn.Module):
    """Routeur adaptatif pour texte - détecte l'importance sémantique des tokens"""

    def __init__(self, dim: int, temperature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.temperature = temperature

        # Détecteurs d'importance pour le langage
        self.semantic_detector = nn.Linear(dim, 1)  # Importance sémantique
        self.syntactic_detector = nn.Conv1d(dim, 1, kernel_size=3, padding=1)  # Structure syntaxique
        self.context_detector = nn.Conv1d(dim, 1, kernel_size=5, padding=2)  # Contexte large

        # Poids apprenables pour chaque expert
        self.expert_weights = nn.Parameter(torch.tensor([3.0, 2.0, 1.0]))

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        x: [batch, seq_len, dim]
        attention_mask: [batch, seq_len] - masque pour les tokens padding
        """
        B, N, C = x.shape

        # Calcul des scores d'importance
        semantic_score = torch.sigmoid(self.semantic_detector(x)).squeeze(-1)  # [B, N]

        x_t = x.transpose(1, 2)  # [B, C, N] pour les convolutions
        syntactic_score = torch.sigmoid(self.syntactic_detector(x_t)).squeeze(1)  # [B, N]
        context_score = torch.sigmoid(self.context_detector(x_t)).squeeze(1)  # [B, N]

        # Appliquer le masque d'attention si fourni
        if attention_mask is not None:
            semantic_score = semantic_score * attention_mask
            syntactic_score = syntactic_score * attention_mask
            context_score = context_score * attention_mask

        # Calcul des masques pour chaque expert
        # Tokens peu importants → Peripheral (ponctuation, mots de liaison)
        peripheral_mask = (1 - semantic_score) * (1 - syntactic_score * 0.5)

        # Tokens moyennement importants → Focal (verbes, adjectifs)
        focal_mask = syntactic_score * (1 - semantic_score * 0.7)

        # Tokens très importants → Reflective (entités, mots-clés)
        reflective_mask = semantic_score * context_score

        # Normalisation
        total = peripheral_mask + focal_mask + reflective_mask + 1e-8
        peripheral_mask = peripheral_mask / total
        focal_mask = focal_mask / total
        reflective_mask = reflective_mask / total

        # Calcul des poids d'utilisation moyens
        usage = torch.stack([
            reflective_mask.mean(),
            focal_mask.mean(),
            peripheral_mask.mean()
        ])

        # Softmax avec température pour contrôler la netteté du routage
        weights = F.softmax(self.expert_weights * usage / self.temperature, dim=0)

        return {
            'peripheral': peripheral_mask,
            'focal': focal_mask,
            'reflective': reflective_mask,
            'weights': weights,
            'usage': usage  # Pour monitoring
        }


class SparseAttentionExpert(nn.Module):
    """Expert d'attention sparse avec k valeurs dynamiques"""

    def __init__(self, dim: int, heads: int, k_keep: int, dropout: float = 0.1, use_bias: bool = False):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.k_keep = k_keep
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.use_bias = use_bias

        # Projections Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=use_bias)
        self.proj = nn.Linear(dim, dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [batch, seq_len, dim]
        mask: [batch, seq_len] - importance mask pour cet expert
        attention_mask: [batch, seq_len] - masque padding
        """
        B, N, C = x.shape

        # Projection Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]

        # Calcul des scores d'attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # Masquage causal pour LLM (empêche de voir le futur)
        causal_mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Masquage des tokens padding si fourni
        if attention_mask is not None:
            # attention_mask: [B, N] -> [B, 1, 1, N]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()
            scores.masked_fill_(~attention_mask, float('-inf'))

        # Sélection sparse des top-k pour chaque position
        if self.k_keep < N:
            # Obtenir les top-k scores pour chaque position de requête
            topk_scores, topk_indices = torch.topk(scores, min(self.k_keep, N), dim=-1)

            # Créer un masque sparse
            sparse_mask = torch.zeros_like(scores, dtype=torch.bool)
            sparse_mask.scatter_(-1, topk_indices, True)

            # Appliquer le masque sparse
            scores.masked_fill_(~sparse_mask, float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Appliquer l'attention
        out = torch.matmul(attn_weights, v)  # [B, H, N, D]
        out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, C]

        # Projection finale avec pondération par le masque d'importance
        out = self.proj(out)
        out = out * mask.unsqueeze(-1)  # Pondération par l'importance

        return out


class AdaptiveAttentionMoE(nn.Module):
    """Module d'attention avec Mixture of Experts adaptatif pour LLM"""

    def __init__(self,
                 dim: int = 768,
                 heads: int = 12,
                 dropout: float = 0.1,
                 temperature: float = 1.0,
                 k_peripheral: int = 32,
                 k_focal: int = 64,
                 k_reflective: int = 128,
                 use_bias: bool = False):
        super().__init__()
        self.dim = dim
        self.heads = heads

        # Routeur adaptatif
        self.router = LanguageFeatureRouter(dim, temperature)

        # Trois experts avec complexité croissante
        # Note: On adapte le nombre de têtes pour chaque expert
        self.peripheral_expert = SparseAttentionExpert(
            dim, max(1, heads // 4), k_keep=k_peripheral, dropout=dropout, use_bias=use_bias
        )
        self.focal_expert = SparseAttentionExpert(
            dim, max(1, heads // 2), k_keep=k_focal, dropout=dropout, use_bias=use_bias
        )
        self.reflective_expert = SparseAttentionExpert(
            dim, heads, k_keep=k_reflective, dropout=dropout, use_bias=use_bias
        )

        # Fusion des sorties
        self.fusion = nn.Linear(dim * 3, dim, bias=use_bias)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x: [batch, seq_len, dim]
        attention_mask: [batch, seq_len]
        Returns: (output, routing_info)
        """
        # Routage adaptatif
        routing = self.router(x, attention_mask)

        # Application des experts
        out_peripheral = self.peripheral_expert(
            x, routing['peripheral'], attention_mask
        )
        out_focal = self.focal_expert(
            x, routing['focal'], attention_mask
        )
        out_reflective = self.reflective_expert(
            x, routing['reflective'], attention_mask
        )

        # Fusion pondérée
        out_combined = torch.cat([
            out_peripheral,
            out_focal,
            out_reflective
        ], dim=-1)

        # Projection finale
        output = self.fusion(out_combined)
        output = self.layer_norm(output)

        return output, routing