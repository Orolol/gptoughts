"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

"""
GPT Language Model implementation with Flash Attention 2 support.
"""

import inspect
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW

# Try to import flash attention 2, but don't fail if not available
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as flash_attn_func
    from flash_attn.bert_padding import unpad_input, pad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Flash Attention 2 not available, falling back to standard attention")

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
        
        # Flash Attention setup
        self.flash = False
        if FLASH_ATTN_AVAILABLE and torch.cuda.is_available():
            try:
                from flash_attn import flash_attn_func
                self.flash = True
                self.flash_fn = flash_attn_func
            except ImportError:
                print("Flash Attention not available")
        
        # ALiBi positional bias - ajusté pour GQA
        self.alibi = AlibiPositionalBias(self.n_head, config.block_size)
        
        # Préallouer le masque causal
        mask = torch.full((config.block_size, config.block_size), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer('mask', mask)

    def forward(self, x, key_value=None, is_generation=False):
        """
        Forward pass avec gestion spécifique pour la génération
        
        Args:
            x: Tensor d'entrée
            key_value: Tensor optionnel pour cross-attention
            is_generation: Bool indiquant si on est en mode génération
        """
        B, T, C = x.size()
        
        # Sauvegarder l'état initial de flash attention
        initial_flash_state = self.flash
        
        # Convertir en bfloat16 pour Flash Attention
        orig_dtype = x.dtype
        if self.flash and x.dtype not in [torch.bfloat16, torch.float16]:
            x = x.to(torch.bfloat16)
        
        try:
            # Si key_value est fourni, on fait de la cross-attention
            if key_value is not None:
                # Projeter la query depuis x
                q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                
                # Projeter key et value depuis key_value
                k = self.k_proj(key_value)
                v = self.v_proj(key_value)
                
                # Reshape pour l'attention
                k = k.view(B, key_value.size(1), self.n_head_kv, self.head_dim).transpose(1, 2)
                v = v.view(B, key_value.size(1), self.n_head_kv, self.head_dim).transpose(1, 2)
                
                # Désactiver Flash Attention pendant la génération pour la cross-attention
                if is_generation:
                    self.flash = False
                
                # Repeat K,V pour GQA seulement si pas en génération
                if not is_generation:
                    k = k.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
                    v = v.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
                
                # Pas de masque causal en cross-attention
                mask = None
                
            else:
                # Self-attention standard
                qkv = torch.cat([
                    self.q_proj(x),
                    self.k_proj(x),
                    self.v_proj(x)
                ], dim=-1)
                
                q, k, v = qkv.split([self.n_embd, self.n_head_kv * self.head_dim, self.n_head_kv * self.head_dim], dim=-1)
                
                q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                k = k.view(B, T, self.n_head_kv, self.head_dim).transpose(1, 2)
                v = v.view(B, T, self.n_head_kv, self.head_dim).transpose(1, 2)
                
                if not is_generation:
                    k = k.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
                    v = v.repeat_interleave(self.n_head // self.n_head_kv, dim=1)
                
                mask = self.mask[:T, :T]

            if self.flash:
                try:
                    # S'assurer que q, k, v sont en bfloat16
                    q = q.to(torch.bfloat16)
                    k = k.to(torch.bfloat16)
                    v = v.to(torch.bfloat16)
                    
                    y = self.flash_fn(q, k, v, causal=mask is not None)
                    # Reconvertir au dtype original
                    y = y.to(orig_dtype)
                except Exception as e:
                    print(f"Flash Attention failed: {e}")
                    self.flash = False
            
            if not self.flash:
                # Attention standard optimisée pour la mémoire
                scale = 1.0 / math.sqrt(self.head_dim)
                att = torch.matmul(q, k.transpose(-2, -1)) * scale
                
                # Appliquer le masque et le bias si nécessaire
                if mask is not None:
                    att = att + self.alibi.get_bias(T, x.device).repeat_interleave(
                        self.n_head // self.n_head_kv, dim=0
                    )[:self.n_head]
                    att = att + mask
                
                # Softmax et dropout
                att = F.softmax(att, dim=-1, dtype=torch.float32)
                att = self.attn_dropout(att.to(q.dtype))
                
                # Calcul final
                y = torch.matmul(att, v)
            
            # Reshape final et reconversion au dtype original si nécessaire
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            if y.dtype != orig_dtype:
                y = y.to(orig_dtype)
            
            return self.resid_dropout(self.o_proj(y))
            
        finally:
            # Restaurer l'état initial de flash attention
            self.flash = initial_flash_state

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU activation function
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        x = self.c_proj(hidden)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Simple residual connections sans checkpoint
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
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

        # Optimiser pour H100
        if device_type == 'cuda':
            # Activer TF32
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Optimisations CUDA
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            
            # Compiler avec des options optimisées pour H100
            if hasattr(torch, 'compile'):
                self = torch.compile(
                    self,
                    mode='max-autotune',  # Utiliser le mode max-autotune au lieu des options détaillées
                    fullgraph=True
                )
                print("Model compiled with max-autotune mode for H100")

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
        B, T = idx.size()
        
        # Forcer le mode d'évaluation
        self.eval()
        
        for _ in range(max_new_tokens):
            # Si la séquence est trop longue, on la tronque
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            with torch.no_grad():
                logits, _ = self.forward(idx_cond)
                logits = logits[:, -1, :] / temperature  # Prendre seulement le dernier token
                
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            
            # Ajouter le nouveau token
            idx = torch.cat([idx, idx_next], dim=1)
        
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

    def forward(self, idx, targets=None):
        """
        Forward pass de l'architecture encoder-decoder.
        
        Args:
            encoder_idx: Tensor d'indices pour l'encodeur [batch_size, encoder_seq_len]
            decoder_idx: Tensor d'indices pour le décodeur [batch_size, decoder_seq_len]
            targets: Tensor cible optionnel pour le calcul de la loss
        """
        
        encoder_idx = idx
        decoder_idx = idx
        
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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.decoder.lm_head(x[:, [-1], :])
            loss = None
            
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure l'optimiseur pour l'ensemble du modèle encoder-decoder.
        Utilise AdamW avec weight decay sur les paramètres appropriés.
        """
        # Collecter tous les paramètres
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Séparer les paramètres qui doivent avoir du weight decay de ceux qui n'en ont pas
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
        
        # Utiliser fused Adam si disponible sur CUDA
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        
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
        B, T = idx.size()
        
        # Forcer le mode d'évaluation
        self.eval()
        
        for _ in range(max_new_tokens):
            # Si la séquence est trop longue, on la tronque
            idx_cond = idx if idx.size(1) <= self.decoder.config.block_size else idx[:, -self.decoder.config.block_size:]
            
            # Forward pass
            with torch.no_grad():
                logits, _ = self.forward(idx_cond)
                logits = logits[:, -1, :] / temperature  # Prendre seulement le dernier token
                
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            
            # Ajouter le nouveau token
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx
