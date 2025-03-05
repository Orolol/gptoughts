"""Attention mechanisms for transformer models."""

import math
import traceback
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func_2
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False

try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

ATTENTION_BACKENDS = {
    'flash_attn_2': FLASH_ATTENTION_AVAILABLE,
    'xformers': XFORMERS_AVAILABLE,
    'sdpa': hasattr(F, 'scaled_dot_product_attention'),
    'standard': True
}

def get_best_attention_backend():
    if ATTENTION_BACKENDS['flash_attn_2']:
        return 'flash_attn_2'
    elif ATTENTION_BACKENDS['xformers']:
        return 'xformers'
    elif ATTENTION_BACKENDS['sdpa']:
        return 'sdpa'
    else:
        return 'standard'

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Grouped-Query Attention (GQA)
        self.n_head = config.n_head
        self.n_head_kv = config.n_head // config.ratio_kv
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # Projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_head_kv * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_head_kv * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Attention backend setup
        if hasattr(config, 'attention_backend') and config.attention_backend is not None:
            if config.attention_backend not in ATTENTION_BACKENDS:
                raise ValueError(f"Attention backend {config.attention_backend} not available")
            if not ATTENTION_BACKENDS[config.attention_backend]:
                print(f"Warning: {config.attention_backend} not available, falling back to best available backend")
                self.attention_backend = get_best_attention_backend()
            else:
                self.attention_backend = config.attention_backend
        else:
            self.attention_backend = get_best_attention_backend()
            
        print(f"Using attention backend: {self.attention_backend}")
        
        # For RoPE positioning
        from .positional_encoding import RoPE
        self.rope = RoPE(self.head_dim, config.block_size)
        
        # Préallouer le masque causal
        mask = torch.full((config.block_size, config.block_size), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer('mask', mask)

    def _memory_efficient_attention(self, q, k, v, mask=None, is_causal=True):
        """
        Wrapper pour l'attention memory efficient avec gestion des erreurs
        """
        try:
            # Ensure all tensors have the same dtype
            working_dtype = torch.float16 if q.device.type == 'cuda' else torch.float32
            q = q.to(working_dtype)
            k = k.to(working_dtype)
            v = v.to(working_dtype)
            
            # Préparer les tenseurs avec memory pinning pour un transfert plus rapide
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            
            # Gérer l'expansion pour GQA si nécessaire
            if k.size(1) != q.size(1):  # Si les dimensions ne correspondent pas (cas GQA)
                # Expand k et v pour correspondre au nombre de têtes de q
                k = k.expand(-1, q.size(1), -1, -1)
                v = v.expand(-1, q.size(1), -1, -1)
            
            # Reshape pour xformers [B, H, T, D] -> [B, T, H, D]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Utiliser un masque optimisé pour xformers
            if is_causal:
                attn_bias = xops.LowerTriangularMask()
            else:
                # Convertir le masque en bias d'attention pour xformers si nécessaire
                attn_bias = xops.AttentionBias.from_mask(mask) if mask is not None else None
            
            # Utiliser la mise en cache des KV pour la génération
            if hasattr(self, '_cached_k') and hasattr(self, '_cached_v'):
                k = torch.cat([self._cached_k, k], dim=1)
                v = torch.cat([self._cached_v, v], dim=1)
                # Ensure cached tensors have the same dtype
                k = k.to(working_dtype)
                v = v.to(working_dtype)
                self._cached_k = k
                self._cached_v = v
            
            y = xops.memory_efficient_attention(
                q, k, v,
                attn_bias=attn_bias,
                p=self.attn_dropout.p if self.training else 0.0,
                scale=1.0 / math.sqrt(self.head_dim),
                op=None  # Let xformers choose the best operator
            )
            
            # Reshape back [B, T, H, D] -> [B, H, T, D]
            return y.transpose(1, 2)
            
        except Exception as e:
            print(f"xformers memory efficient attention failed: {e}")
            print(f"Tensor dtypes - q: {q.dtype}, k: {k.dtype}, v: {v.dtype}")
            return None

    def _flash_attention(self, q, k, v, mask=None, is_causal=True):
        """
        Wrapper pour Flash Attention 2 avec optimisations
        """
        try:
            # Convert inputs to bfloat16
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
            
            # Handle GQA by expanding k and v heads
            if k.size(1) != q.size(1):  # If number of heads don't match (GQA case)
                # Repeat k and v heads to match q heads
                k = k.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
                v = v.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
            
            # Utiliser la mise en cache des KV pour la génération
            if hasattr(self, '_cached_k') and hasattr(self, '_cached_v'):
                k = torch.cat([self._cached_k, k], dim=2)  # dim=2 car Flash Attention utilise [B, H, T, D]
                v = torch.cat([self._cached_v, v], dim=2)
                self._cached_k = k
                self._cached_v = v
            
            with torch.amp.autocast(enabled=True, device_type='cuda'):
                # Appliquer Flash Attention avec les optimisations
                output = flash_attn_func_2(
                    q, k, v,
                    causal=is_causal,
                    softmax_scale=1.0 / math.sqrt(self.head_dim)
                )
            
            return output
            
        except Exception as e:
            print(f"Flash Attention 2 failed: {e}")
            print(f"Input dtypes - q: {q.dtype}, k: {k.dtype}, v: {v.dtype}")
            print(f"Input shapes - q: {q.shape}, k: {k.shape}, v: {v.shape}")
            return None

    def _sdpa_attention(self, q, k, v, mask=None, is_causal=True):
        """
        Wrapper pour Scaled Dot Product Attention avec optimisations
        """
        try:
            # Utiliser la mise en cache des KV pour la génération
            if hasattr(self, '_cached_k') and hasattr(self, '_cached_v'):
                k = torch.cat([self._cached_k, k], dim=2)
                v = torch.cat([self._cached_v, v], dim=2)
                self._cached_k = k
                self._cached_v = v
            
            # Utiliser SDPA avec optimisations
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None if is_causal else mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
                scale=1.0 / math.sqrt(self.head_dim)
            )
        except Exception as e:
            print(f"SDPA failed: {e}")
            return None

    def _standard_attention(self, q, k, v, mask=None, is_causal=True):
        """
        Implementation standard de l'attention avec optimisations mémoire
        """
        # Utiliser la mise en cache des KV pour la génération
        if hasattr(self, '_cached_k') and hasattr(self, '_cached_v'):
            k = torch.cat([self._cached_k, k], dim=2)
            v = torch.cat([self._cached_v, v], dim=2)
            self._cached_k = k
            self._cached_v = v
        
        # Calculer l'attention avec optimisations mémoire
        scale = 1.0 / math.sqrt(self.head_dim)
        
        # Utiliser torch.baddbmm pour une multiplication matricielle plus efficace
        att = torch.empty(q.shape[:-2] + (q.shape[-2], k.shape[-2]), 
                         dtype=q.dtype, device=q.device)
        att = torch.baddbmm(
            att, q, k.transpose(-2, -1),
            beta=0, alpha=scale
        )
        
        # Clipping pour stabilité numérique
        att = torch.clamp(att, min=-1e4, max=1e4)
        
        # Appliquer le masque causal si nécessaire
        if is_causal:
            if mask is not None:
                att = att + mask
        
        # Utiliser float32 pour le softmax pour plus de stabilité
        att = F.softmax(att.float(), dim=-1).to(q.dtype)
        att = self.attn_dropout(att)
        
        # Utiliser torch.baddbmm pour le produit final
        out = torch.empty(att.shape[:-2] + (att.shape[-2], v.shape[-1]), 
                         dtype=q.dtype, device=q.device)
        out = torch.bmm(att, v)
        
        return out

    def forward(self, x, key_value=None, is_generation=False):
        B, T, C = x.size()
        
        # Ensure consistent dtype for all tensors
        working_dtype = torch.float16 if x.device.type == 'cuda' else torch.float32
        x = x.to(working_dtype)
        
        try:
            # Vérification et correction des NaN/Inf
            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            
            # Cross-attention
            if key_value is not None:
                q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                k = self.k_proj(key_value)
                v = self.v_proj(key_value)
                
                k = k.view(B, key_value.size(1), self.n_head_kv, self.head_dim).transpose(1, 2)
                v = v.view(B, key_value.size(1), self.n_head_kv, self.head_dim).transpose(1, 2)
                
                # Convert k and v to the same dtype as q
                k = k.to(working_dtype)
                v = v.to(working_dtype)
                
                # Répéter K,V pour GQA
                k = k.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
                v = v.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
                
                # Apply RoPE to queries and keys
                q = self.rope(q)
                k = self.rope(k)
                
            else:
                # Self-attention
                qkv = torch.cat([
                    self.q_proj(x),
                    self.k_proj(x),
                    self.v_proj(x)
                ], dim=-1)
                
                # Ensure consistent dtype
                qkv = qkv.to(working_dtype)
                
                q, k, v = qkv.split([self.n_embd, self.n_head_kv * self.head_dim, self.n_head_kv * self.head_dim], dim=-1)
                
                q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                k = k.view(B, T, self.n_head_kv, self.head_dim).transpose(1, 2)
                v = v.view(B, T, self.n_head_kv, self.head_dim).transpose(1, 2)
                
                # Répéter K,V pour GQA si ce n'est pas de la génération
                if not is_generation:
                    k = k.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
                    v = v.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
                
                # Apply RoPE to queries and keys
                q = self.rope(q)
                k = self.rope(k)

            # Déterminer si nous sommes en mode causal et préparer le masque
            is_causal = key_value is None  # Causal seulement pour self-attention
            attn_mask = self.mask[:T, :T] if is_causal else None
            
            y = None
            # During generation or cross-attention, use SDPA instead of Flash Attention
            if is_generation or key_value is not None:
                y = self._sdpa_attention(q, k, v, attn_mask, is_causal)
            else:
                # Try Flash Attention first
                if self.attention_backend == 'flash_attn_2':
                    y = self._flash_attention(q, k, v, attn_mask, is_causal)
            
            if y is None and self.attention_backend == 'xformers':
                y = self._memory_efficient_attention(q, k, v, attn_mask, is_causal)
            
            if y is None and self.attention_backend == 'sdpa':
                y = self._sdpa_attention(q, k, v, attn_mask, is_causal)
            
            if y is None:
                # Standard attention
                scale = 1.0 / math.sqrt(self.head_dim)
                att = torch.matmul(q, k.transpose(-2, -1)) * scale
                
                # Clipping pour stabilité numérique
                att = torch.clamp(att, min=-1e4, max=1e4)
                
                # Appliquer le masque causal si nécessaire
                if is_causal and attn_mask is not None:
                    att = att + attn_mask
                
                # Utiliser float32 pour le softmax pour plus de stabilité
                att = F.softmax(att.float(), dim=-1).to(working_dtype)
                att = self.attn_dropout(att)
                
                y = torch.matmul(att, v)
            
            # Vérification finale des NaN/Inf
            if torch.isnan(y).any() or torch.isinf(y).any():
                y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
            
            # Reshape et projection finale
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            
            # Projection finale
            output = self.resid_dropout(self.o_proj(y))
            
            return output
            
        except Exception as e:
            print(f"Attention computation failed: {e}")
            print(traceback.format_exc())
            raise  # Re-raise the exception after printing the traceback 