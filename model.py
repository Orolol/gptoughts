import inspect
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
import torch.utils.checkpoint as checkpoint
import traceback

# Configuration des backends d'attention
ATTENTION_BACKENDS = {}

# Try to import flash attention 2
try:
    from flash_attn import flash_attn_func as flash_attn_func_2
    ATTENTION_BACKENDS['flash_attn_2'] = True
    print("Flash Attention 2 available")
except ImportError:
    ATTENTION_BACKENDS['flash_attn_2'] = False
    print("Flash Attention 2 not available")

# Try to import xformers with better error handling
ATTENTION_BACKENDS['xformers'] = False
try:
    import xformers
    import xformers.ops as xops
    # Vérifier si les opérations d'attention sont réellement disponibles
    if hasattr(xops, 'memory_efficient_attention'):
        ATTENTION_BACKENDS['xformers'] = True
        print("xformers memory efficient attention available")
    else:
        print("xformers installed but memory efficient attention not available")
except ImportError as e:
    print(f"xformers not available: {e}")
except Exception as e:
    print(f"Error loading xformers: {e}")

# Check if SDPA with CUDA backend is available
ATTENTION_BACKENDS['sdpa'] = hasattr(F, "scaled_dot_product_attention") and torch.cuda.is_available()
if ATTENTION_BACKENDS['sdpa']:
    print("PyTorch SDPA available")

def get_best_attention_backend():
    """
    Retourne le meilleur backend d'attention disponible dans l'ordre de préférence:
    1. Flash Attention 2
    2. xformers
    3. SDPA
    4. None (attention standard)
    """
    if ATTENTION_BACKENDS['flash_attn_2']:
        return 'flash_attn_2'
    elif ATTENTION_BACKENDS['xformers']:
        return 'xformers'
    elif ATTENTION_BACKENDS['sdpa']:
        return 'sdpa'
    return None

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self._norm(x.float()).type_as(x)

class AlibiPositionalBias(nn.Module):
    """
    ALiBi (Attention with Linear Biases) implementation.
    Paper: https://arxiv.org/abs/2108.12409
    """
    def __init__(self, num_heads: int, max_seq_len: int):
        super().__init__()
        
        # Calculate ALiBi slopes
        def get_slopes(n_heads: int) -> torch.Tensor:
            def get_slopes_power_of_2(n_heads: int) -> list:
                start = 2 ** (-(2 ** -(math.log2(n_heads) - 3)))
                ratio = start
                return [start * (ratio ** i) for i in range(n_heads)]

            if math.log2(n_heads).is_integer():
                return torch.tensor(get_slopes_power_of_2(n_heads))
            
            closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
            base_slopes = torch.tensor(get_slopes_power_of_2(closest_power_of_2))
            
            if n_heads <= 2 * closest_power_of_2:
                slopes = torch.concat([base_slopes, base_slopes[0::2]])[:n_heads]
            else:
                slopes = torch.concat([base_slopes, base_slopes[0::2], base_slopes[1::2]])[:n_heads]
            
            return slopes

        slopes = get_slopes(num_heads)
        # Create position bias matrix
        pos = torch.arange(max_seq_len)
        rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)  # [seq_len, seq_len]
        
        # [num_heads, seq_len, seq_len]
        alibi = slopes.unsqueeze(1).unsqueeze(1) * rel_pos.unsqueeze(0)
        
        # Register as buffer since it's position-dependent but not trainable
        self.register_buffer('alibi', alibi)
        
    def get_bias(self, T: int, device: torch.device) -> torch.Tensor:
        return self.alibi.to(device)[:, :T, :T]

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
                raise ValueError(f"Attention backend {config.attention_backend} not available. Available backends: {list(ATTENTION_BACKENDS.keys())}")
            if not ATTENTION_BACKENDS[config.attention_backend]:
                raise ValueError(f"Attention backend {config.attention_backend} is not installed or not working properly")
            self.attention_backend = config.attention_backend
        else:
            self.attention_backend = get_best_attention_backend()
            
        print(f"Using attention backend: {self.attention_backend}")
        
        # ALiBi positional bias - ajusté pour GQA
        self.alibi = AlibiPositionalBias(self.n_head, config.block_size)
        
        # Préallouer le masque causal
        mask = torch.full((config.block_size, config.block_size), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer('mask', mask)

    def _memory_efficient_attention(self, q, k, v, mask=None, is_causal=True):
        """
        Wrapper pour l'attention memory efficient avec gestion des erreurs
        """
        try:
            # Préparer les tenseurs
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
            
            attn_bias = xops.LowerTriangularMask() if is_causal else None
            
            y = xops.memory_efficient_attention(
                q, k, v,
                attn_bias=attn_bias,
                p=self.attn_dropout.p if self.training else 0.0,
                scale=1.0 / math.sqrt(self.head_dim)
            )
            
            # Reshape back [B, T, H, D] -> [B, H, T, D]
            return y.transpose(1, 2)
            
        except Exception as e:
            print(f"xformers memory efficient attention failed: {e}")
            return None

    def _flash_attention(self, q, k, v, mask=None, is_causal=True):
        """
        Wrapper pour Flash Attention 2
        """
        try:
            return flash_attn_func_2(q, k, v, causal=is_causal)
        except Exception as e:
            print(f"Flash Attention 2 failed: {e}")
            return None

    def _sdpa_attention(self, q, k, v, mask=None, is_causal=True):
        """
        Wrapper pour Scaled Dot Product Attention
        """
        try:
            # Ne pas passer de masque explicite si is_causal=True
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None if is_causal else mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal
            )
        except Exception as e:
            print(f"SDPA failed: {e}")
            return None

    def _standard_attention(self, q, k, v, mask=None, is_causal=True):
        """
        Implementation standard de l'attention avec optimisations mémoire
        """
        scale = 1.0 / math.sqrt(self.head_dim)
        att = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Clipping pour stabilité numérique
        att = torch.clamp(att, min=-1e4, max=1e4)
        
        # Appliquer le masque et le bias
        if is_causal:
            if mask is not None:
                att = att + mask
            att = att + self.alibi.get_bias(q.size(-2), q.device)
        
        # Utiliser float32 pour le softmax pour plus de stabilité
        att = F.softmax(att.float(), dim=-1).to(q.dtype)
        att = self.attn_dropout(att)
        
        return torch.matmul(att, v)

    def forward(self, x, key_value=None, is_generation=False):
        B, T, C = x.size()
        
        # Convertir en bfloat16 si nécessaire pour Flash Attention 2
        orig_dtype = x.dtype
        if self.attention_backend == 'flash_attn_2' and x.dtype not in [torch.bfloat16, torch.float16]:
            x = x.to(torch.bfloat16)
        
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
                
                # Répéter K,V pour GQA
                k = k.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
                v = v.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
                
                # Pendant la génération, augmenter l'influence de la cross-attention
                if is_generation:
                    # Augmenter l'importance des requêtes et des clés
                    q = q * 2.0
                    k = k * 2.0
                    
                    # Ajouter un biais positif pour favoriser l'attention sur le prompt
                    prompt_bias = torch.zeros((B, self.n_head, T, key_value.size(1)), device=x.device)
                    prompt_bias = prompt_bias + 0.5  # Biais positif
                    
            else:
                # Self-attention
                qkv = torch.cat([
                    self.q_proj(x),
                    self.k_proj(x),
                    self.v_proj(x)
                ], dim=-1)
                
                q, k, v = qkv.split([self.n_embd, self.n_head_kv * self.head_dim, self.n_head_kv * self.head_dim], dim=-1)
                
                q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                k = k.view(B, T, self.n_head_kv, self.head_dim).transpose(1, 2)
                v = v.view(B, T, self.n_head_kv, self.head_dim).transpose(1, 2)
                
                # Répéter K,V pour GQA si ce n'est pas de la génération
                if not is_generation:
                    k = k.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
                    v = v.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
                    prompt_bias = None
                else:
                    prompt_bias = None
            
            # Déterminer si nous sommes en mode causal et préparer le masque
            is_causal = key_value is None  # Causal seulement pour self-attention
            attn_mask = self.mask[:T, :T] if is_causal else None
            
            y = None
            if self.attention_backend == 'flash_attn_2':
                y = self._flash_attention(q, k, v, attn_mask, is_causal)
            
            if y is None and self.attention_backend == 'xformers':
                y = self._memory_efficient_attention(q, k, v, attn_mask, is_causal)
            
            if y is None and self.attention_backend == 'sdpa':
                y = self._sdpa_attention(q, k, v, attn_mask, is_causal)
            
            if y is None:
                # Attention standard avec biais de prompt
                scale = 1.0 / math.sqrt(self.head_dim)
                att = torch.matmul(q, k.transpose(-2, -1)) * scale
                
                # Ajouter le biais de prompt si présent
                if prompt_bias is not None:
                    att = att + prompt_bias
                
                # Clipping pour stabilité numérique
                att = torch.clamp(att, min=-1e4, max=1e4)
                
                # Appliquer le masque et le bias
                if is_causal:
                    if attn_mask is not None:
                        att = att + attn_mask
                    att = att + self.alibi.get_bias(q.size(-2), q.device)
                
                # Utiliser float32 pour le softmax pour plus de stabilité
                att = F.softmax(att.float(), dim=-1).to(q.dtype)
                att = self.attn_dropout(att)
                
                y = torch.matmul(att, v)
            
            # Vérification finale des NaN/Inf
            if torch.isnan(y).any() or torch.isinf(y).any():
                y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
            
            # Reshape et projection finale
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            
            # Reconvertir au dtype original si nécessaire
            if y.dtype != orig_dtype:
                y = y.to(orig_dtype)
            
            return self.resid_dropout(self.o_proj(y))
            
        except Exception as e:
            print(f"Attention computation failed: {e}")
            print(traceback.format_exc())
            # Fallback sur l'attention standard en cas d'erreur
            return self._standard_attention(q, k, v, attn_mask, is_causal)

class MLP(nn.Module):
    """
    Multi-Layer Perceptron avec activation SwiGLU optimisée et gestion efficace de la mémoire.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Utiliser 4 * n_embd comme dimension cachée par défaut
        hidden_dim = 4 * config.n_embd
        
        # Projections combinées pour réduire le nombre d'opérations
        self.gate_up_proj = nn.Linear(config.n_embd, 2 * hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Activation function setup
        self.act_fn = self._get_optimized_activation()
        
        # Initialize weights with a special scale for better gradient flow
        self._init_weights()

    def _init_weights(self):
        """
        Initialisation spéciale des poids pour une meilleure convergence
        """
        # Scaled initialization for better gradient flow
        scale = 2 / (self.config.n_embd ** 0.5)
        
        nn.init.normal_(self.gate_up_proj.weight, mean=0.0, std=scale)
        if self.gate_up_proj.bias is not None:
            nn.init.zeros_(self.gate_up_proj.bias)
            
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=scale)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)

    def _get_optimized_activation(self):
        """
        Retourne la fonction d'activation optimisée selon le device et les capacités
        """
        try:
            # Vérifier si on peut utiliser une implémentation CUDA optimisée
            if torch.cuda.is_available() and hasattr(torch.nn.functional, 'silu'):
                def optimized_swiglu(x):
                    # Éviter les opérations inplace sur les vues
                    x1, x2 = x.chunk(2, dim=-1)
                    return F.silu(x2) * x1
                return optimized_swiglu
        except:
            pass
        
        # Fallback sur l'implémentation standard
        def standard_swiglu(x):
            x1, x2 = x.chunk(2, dim=-1)
            return F.silu(x2) * x1
        return standard_swiglu

    def _fuse_operations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fusionne les opérations quand c'est possible pour une meilleure efficacité
        """
        # Combiner les projections up et gate en une seule opération
        combined = self.gate_up_proj(x)
        
        # Appliquer l'activation
        hidden = self.act_fn(combined)
        
        # Projection finale avec dropout
        return self.dropout(self.down_proj(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass avec gestion optimisée de la mémoire
        """
        if torch.jit.is_scripting() or not self.training:
            # Mode inference ou JIT: utiliser l'implémentation fusionnée
            return self._fuse_operations(x)
        
        # Mode training: utiliser la version avec checkpointing si la séquence est longue
        if x.shape[1] > 1024:  # Seuil arbitraire, peut être ajusté
            return checkpoint.checkpoint(self._fuse_operations, x, use_reentrant=False)
        
        return self._fuse_operations(x)

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)
        self.use_checkpoint = True

    def _attn_block(self, x, key_value=None):
        ln_out = self.ln_1(x)
        if key_value is not None:
            return self.attn(ln_out, key_value=key_value)
        return self.attn(ln_out)

    def _mlp_block(self, x):
        return self.mlp(self.ln_2(x))

    def forward(self, x, key_value=None):
        # Disable gradient checkpointing when using compiled model
        if hasattr(self, '_compiled_forward'):
            self.use_checkpoint = False
        
        if self.use_checkpoint and self.training:
            # Modified wrapper for gradient checkpointing
            def create_custom_forward(func):
                def custom_forward(*inputs):
                    # Filter out None inputs
                    valid_inputs = [inp for inp in inputs if inp is not None]
                    
                    # Don't modify requires_grad dynamically
                    return func(*valid_inputs)
                return custom_forward

            # Attention with checkpoint
            attn_func = create_custom_forward(self._attn_block)
            attn_out = checkpoint.checkpoint(
                attn_func, 
                x, 
                key_value,
                use_reentrant=False,
                preserve_rng_state=True
            )
            x = x + attn_out

            # MLP with checkpoint
            mlp_func = create_custom_forward(self._mlp_block)
            mlp_out = checkpoint.checkpoint(
                mlp_func, 
                x,
                use_reentrant=False,
                preserve_rng_state=True
            )
            x = x + mlp_out
        else:
            # Standard forward pass without checkpoint
            x = x + self._attn_block(x, key_value)
            x = x + self._mlp_block(x)

        return x

@dataclass
class GPTConfig:
    """Configuration class for GPT model hyperparameters."""
    
    block_size: int = 1024  # Maximum sequence length
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64
    n_layer: int = 12       # Number of transformer layers
    n_head: int = 12        # Number of attention heads
    n_embd: int = 768      # Embedding dimension
    dropout: float = 0.0    # Dropout probability
    bias: bool = True       # Whether to use bias in Linear and LayerNorm layers
    ratio_kv: int = 4      # Ratio of key/value heads
    label_smoothing: float = 0.1  # Label smoothing factor
    attention_backend: Optional[str] = None  # Force specific attention backend (flash_attn_2, xformers, sdpa, or None for auto)

class GPT(nn.Module):
    def __init__(self, config, embedding_layer, pos_embedding_layer):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        
        # Stocker les références aux embeddings partagés
        self.wte = embedding_layer
        self.wpe = pos_embedding_layer
        
        # Créer le lm_head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        self.gradient_accumulation_steps = 1

    def configure_training(self, batch_size, device_memory_gb):
        """Configure optimal batch size and gradient accumulation"""
        # Estimation grossière basée sur la mémoire GPU disponible
        params_size = self.get_num_params() * 4  # 4 bytes par paramètre
        activation_size = batch_size * self.config.block_size * self.config.n_embd * 4
        
        # Ajuster le batch size et gradient accumulation en fonction de la mémoire
        total_memory = device_memory_gb * 1e9
        optimal_batch_size = min(
            batch_size,
            int((total_memory * 0.7 - params_size) / activation_size)
        )
        
        self.gradient_accumulation_steps = max(1, batch_size // optimal_batch_size)
        
        return optimal_batch_size, self.gradient_accumulation_steps

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wpe'):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            
            if self.config.label_smoothing > 0.0:
                # Calculer la loss avec label smoothing
                num_classes = logits.size(-1)
                smoothing = self.config.label_smoothing
                
                # Créer les labels lissés
                confidence = 1.0 - smoothing
                smoothing_value = smoothing / (num_classes - 1)
                
                # One-hot avec label smoothing
                true_dist = torch.zeros_like(logits)
                true_dist.fill_(smoothing_value)
                true_dist.scatter_(
                    -1, 
                    targets.unsqueeze(-1), 
                    confidence
                )
                
                # Calculer la KL divergence
                log_probs = F.log_softmax(logits.view(-1, logits.size(-1)), dim=-1)
                loss = -(true_dist.view(-1, true_dist.size(-1)) * log_probs).sum(-1)
                
                # Masquer les positions de padding
                mask = (targets != -1).float()
                loss = (loss * mask.view(-1)).sum() / mask.sum()
            else:
                # Loss standard sans label smoothing
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @torch.no_grad()
    def generate(self, idx: torch.LongTensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.LongTensor:
        """
        Generate text tokens autoregressively.

        Args:
            idx: Conditioning sequence of indices (shape: batch_size × sequence_length)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (higher = more random, lower = more deterministic)
            top_k: If set, only sample from the top k most likely tokens

        Returns:
            torch.LongTensor: Generated sequence including the conditioning tokens
            
        Note:
            Make sure the model is in eval() mode before generation.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @staticmethod
    def configure_dataloader(dataset, batch_size, num_workers=None):
        if num_workers is None:
            num_workers = min(8, os.cpu_count() // 2)  # Heuristique raisonnable
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )

    def set_gradient_checkpointing(self, value: bool):
        """Set gradient checkpointing for all transformer blocks."""
        for block in self.transformer.h:
            block.use_checkpoint = value

class EncoderDecoderGPT(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        
        # S'assurer que les dimensions d'embedding sont compatibles
        assert encoder_config.n_embd == decoder_config.n_embd
        assert encoder_config.vocab_size == decoder_config.vocab_size
        assert encoder_config.block_size == decoder_config.block_size
        
        # Créer les embeddings partagés
        self.shared_embedding = nn.Embedding(encoder_config.vocab_size, encoder_config.n_embd)
        self.shared_pos_embedding = nn.Embedding(encoder_config.block_size, encoder_config.n_embd)
        
        # Créer l'encodeur et le décodeur avec les embeddings partagés
        self.encoder = GPT(encoder_config, 
                         embedding_layer=self.shared_embedding,
                         pos_embedding_layer=self.shared_pos_embedding)
        self.decoder = GPT(decoder_config, 
                         embedding_layer=self.shared_embedding,
                         pos_embedding_layer=self.shared_pos_embedding)
        
        # Partager avec la couche de sortie du décodeur (weight tying)
        self.decoder.lm_head.weight = self.shared_embedding.weight
        
        # Cross-attention et layer norms
        self.cross_attention = nn.ModuleList([
            CausalSelfAttention(decoder_config) 
            for _ in range(decoder_config.n_layer)
        ])
        
        self.cross_ln = nn.ModuleList([
            RMSNorm(decoder_config.n_embd)
            for _ in range(decoder_config.n_layer)
        ])
        
        # Cache pour les états de l'encodeur
        self._cached_encoder_output = None
        self._cached_encoder_input = None

    def forward(self, encoder_idx, decoder_idx, targets=None):
        """
        Forward pass de l'architecture encoder-decoder.
        
        Args:
            encoder_idx: Tensor d'indices pour l'encodeur [batch_size, encoder_seq_len]
            decoder_idx: Tensor d'indices pour le décodeur [batch_size, decoder_seq_len]
            targets: Tensor cible optionnel pour le calcul de la loss
        """
        # Vérifier les dimensions des entrées
        if encoder_idx.dim() == 4:
            encoder_idx = encoder_idx.squeeze(0)
            
        encoder_seq_len = encoder_idx.size(1)
        decoder_seq_len = decoder_idx.size(1)
        
        # Vérifier que les séquences ne dépassent pas block_size
        encoder_seq_len = min(encoder_seq_len, self.encoder.config.block_size)
        decoder_seq_len = min(decoder_seq_len, self.decoder.config.block_size)
        
        encoder_idx = encoder_idx[:, :encoder_seq_len]
        decoder_idx = decoder_idx[:, :decoder_seq_len]
        
        # Ajouter des vérifications de NaN après chaque étape majeure
        def check_and_fix_nans(tensor, name):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"NaN/Inf detected in {name}")
                return torch.nan_to_num(tensor, nan=0.0, posinf=1e4, neginf=-1e4)
            return tensor
        
        # Encoder forward pass
        with torch.no_grad():
            # Générer les positions pour l'encodeur
            encoder_pos = torch.arange(0, encoder_seq_len, dtype=torch.long, device=encoder_idx.device)
            
            # Forward pass de l'encodeur
            tok_emb = self.shared_embedding(encoder_idx)  # [batch_size, seq_len, n_embd]
            pos_emb = self.shared_pos_embedding(encoder_pos)  # [seq_len, n_embd]
            
            # Ajuster les dimensions de pos_emb pour le broadcasting
            pos_emb = pos_emb.unsqueeze(0)  # [1, seq_len, n_embd]
            pos_emb = pos_emb.expand(tok_emb.size(0), -1, -1)  # [batch_size, seq_len, n_embd]
            
            x = self.encoder.transformer.drop(tok_emb + pos_emb)
            
            for block in self.encoder.transformer.h:
                x = block(x)
            
            encoder_hidden = self.encoder.transformer.ln_f(x)
        
        # Decoder forward pass
        # Générer les positions pour le décodeur
        decoder_pos = torch.arange(0, decoder_seq_len, dtype=torch.long, device=decoder_idx.device)
        
        # Forward pass du décodeur
        tok_emb = self.shared_embedding(decoder_idx)  # [batch_size, seq_len, n_embd]
        pos_emb = self.shared_pos_embedding(decoder_pos)  # [seq_len, n_embd]
        
        # Ajuster les dimensions de pos_emb pour le broadcasting
        pos_emb = pos_emb.unsqueeze(0)  # [1, seq_len, n_embd]
        pos_emb = pos_emb.expand(tok_emb.size(0), -1, -1)  # [batch_size, seq_len, n_embd]
        
        x = self.decoder.transformer.drop(tok_emb + pos_emb)
        
        # Appliquer les blocs de décodeur avec cross-attention
        for i, block in enumerate(self.decoder.transformer.h):
            # Self-attention standard
            x = x + block.attn(block.ln_1(x), is_generation=True)
            
            # Cross-attention avec les hidden states de l'encodeur
            cross_x = self.cross_ln[i](x)
            x = x + self.cross_attention[i](
                cross_x, 
                key_value=encoder_hidden,
                is_generation=True
            )
            
            # MLP
            x = x + block.mlp(block.ln_2(x))
        
        x = self.decoder.transformer.ln_f(x)
        
        # Calculer les logits et la loss si nécessaire
        if targets is not None:
            # S'assurer que targets a la bonne taille
            targets = targets[:, :decoder_seq_len]
            logits = self.decoder.lm_head(x)
            
            if self.decoder.config.label_smoothing > 0.0:
                # Calculer la loss avec label smoothing
                num_classes = logits.size(-1)
                smoothing = self.decoder.config.label_smoothing
                
                # Créer les labels lissés
                confidence = 1.0 - smoothing
                smoothing_value = smoothing / (num_classes - 1)
                
                # One-hot avec label smoothing
                true_dist = torch.zeros_like(logits)
                true_dist.fill_(smoothing_value)
                true_dist.scatter_(
                    -1, 
                    targets.unsqueeze(-1), 
                    confidence
                )
                
                # Calculer la KL divergence
                log_probs = F.log_softmax(logits.view(-1, logits.size(-1)), dim=-1)
                loss = -(true_dist.view(-1, true_dist.size(-1)) * log_probs).sum(-1)
                
                # Masquer les positions de padding
                mask = (targets != -1).float()
                loss = (loss * mask.view(-1)).sum() / mask.sum()
            else:
                # Loss standard sans label smoothing
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
            # Ajouter la régularisation pour DDP si en mode training
            if self.training:
                reg_loss = 0
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        reg_loss = reg_loss + 0.0 * param.mean()
                loss = loss + reg_loss
        else:
            logits = self.decoder.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss
    
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure the AdamW optimizer with weight decay.
        
        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            betas: Adam betas parameters
            device_type: Device type ('cuda' or 'cpu')
            
        Returns:
            Configured AdamW optimizer
        """
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        # Utiliser AdaFactor au lieu de AdamW pour une meilleure efficacité mémoire
        try:
            from transformers.optimization import Adafactor
            optimizer = Adafactor(
                self.parameters(),
                lr=learning_rate,
                scale_parameter=True,
                relative_step=False,
                warmup_init=False,
                clip_threshold=1.0
            )
        except ImportError:
            # Fallback sur AdamW optimisé
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer




    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Génère du texte de manière auto-régressive.
        
        Args:
            idx: Tensor d'indices [batch_size, seq_len]
            max_new_tokens: Nombre de nouveaux tokens à générer
            temperature: Température pour l'échantillonnage
            top_k: Nombre de tokens les plus probables à considérer
        """
        # Initialiser le tenseur de sortie du décodeur avec le token de début
        decoder_idx = torch.full((idx.size(0), 1), 
                               self.shared_embedding.weight.size(0)-1,  # EOS token 
                               dtype=torch.long, 
                               device=idx.device)
        
        # Encoder une seule fois la séquence d'entrée
        with torch.no_grad():
            # Préparer l'entrée de l'encodeur
            encoder_seq_len = min(idx.size(1), self.encoder.config.block_size)
            idx = idx[:, :encoder_seq_len]
            
            # Générer les positions pour l'encodeur
            encoder_pos = torch.arange(0, encoder_seq_len, dtype=torch.long, device=idx.device)
            
            # Forward pass de l'encodeur
            tok_emb = self.shared_embedding(idx)
            pos_emb = self.shared_pos_embedding(encoder_pos)
            pos_emb = pos_emb.unsqueeze(0).expand(tok_emb.size(0), -1, -1)
            
            x = self.encoder.transformer.drop(tok_emb + pos_emb)
            for block in self.encoder.transformer.h:
                x = block(x)
        
        # Génération auto-régressive
        for _ in range(max_new_tokens):
            # Limiter la taille de la séquence du décodeur si nécessaire
            if decoder_idx.size(1) > self.decoder.config.block_size:
                decoder_idx = decoder_idx[:, -self.decoder.config.block_size:]
            
            # Forward pass complet
            logits, _ = self(idx, decoder_idx)
            
            # Échantillonnage du prochain token
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Concaténer le nouveau token
            decoder_idx = torch.cat((decoder_idx, idx_next), dim=1)
        
        return decoder_idx

    def set_gradient_checkpointing(self, value: bool):
        """Set gradient checkpointing for both encoder and decoder."""
        # Set checkpointing for encoder blocks
        for block in self.encoder.transformer.h:
            block.use_checkpoint = value
            
        # Set checkpointing for decoder blocks
        for block in self.decoder.transformer.h:
            block.use_checkpoint = value
            
        # Set checkpointing for cross attention blocks
        for block in self.cross_attention:
            if hasattr(block, 'use_checkpoint'):
                block.use_checkpoint = value
