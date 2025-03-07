import math
import time
import inspect
import traceback
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

# Import model config
from models.config import GPTConfig

# Import block components
from models.blocks.normalization import RMSNorm
from models.blocks.positional_encoding import RoPE, AlibiPositionalBias
from models.blocks.attention import CausalSelfAttention, get_best_attention_backend
from models.blocks.mlp import MLP, Block
from models.blocks.moe import Router, ExpertGroup, MoELayer

# Attempt imports for optimized attention implementations
try:
    from flash_attn import flash_attn_func_2
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False

try:
    import xformers
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

class GPT(nn.Module):
    def __init__(self, config, embedding_layer, pos_embedding_layer):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = embedding_layer, 
            wpe = pos_embedding_layer, 
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        
        # Stocker les références aux embeddings partagés
        self.wte = embedding_layer
        self.wpe = pos_embedding_layer
        
        # Créer le lm_head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Set gradient checkpointing for transformer blocks
        for block in self.transformer.h:
            block.use_checkpoint = config.use_gradient_checkpointing
        
        # Tie embeddings and LM head weights
        self.lm_head.weight = self.transformer.wte.weight
        
        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Special scaled init for output projection
        scale = 0.02 / math.sqrt(2 * config.n_layer)
        self.lm_head.weight.data.normal_(mean=0.0, std=scale)
        
        # Total parameter count
        self.param_count = self.get_num_params()
        print(f"Number of parameters: {self.param_count/1e6:.2f}M")

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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, optimizer_type=None):
        """
        Configure optimizer with weight decay.
        
        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            betas: Adam beta parameters
            device_type: Device type ('cuda' or 'cpu')
            optimizer_type: Type of optimizer to use (default: 'adamw')
            
        Returns:
            Configured optimizer
        """
        from models.optimizers import configure_optimizer_for_gpt
        return configure_optimizer_for_gpt(
            model=self,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
            device_type=device_type,
            optimizer_type=optimizer_type
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
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, optimizer_type=None):
        """
        Configure optimizer with weight decay.
        
        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            betas: Adam beta parameters
            device_type: Device type ('cuda' or 'cpu')
            optimizer_type: Type of optimizer to use (default: 'adamw')
            
        Returns:
            Configured optimizer
        """
        from models.optimizers import configure_optimizer_for_gpt
        return configure_optimizer_for_gpt(
            model=self,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
            device_type=device_type,
            optimizer_type=optimizer_type
        )

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
