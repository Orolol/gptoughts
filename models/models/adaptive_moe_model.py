"""
Adaptive MoE Language Model
LLM with MoE applied to attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import math

from ..blocks.adaptive_moe_block import AdaptiveMoEBlock
from ..blocks.normalization import RMSNorm
from ..blocks.positional_encoding import apply_rope, precompute_freqs_cis


@dataclass
class AdaptiveMoEConfig:
    """Configuration for Adaptive MoE Language Model"""
    # Model architecture
    vocab_size: int = 50257
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12
    block_size: int = 2048

    # MoE Attention parameters
    k_peripheral: int = 32
    k_focal: int = 64
    k_reflective: int = 128
    router_temperature: float = 1.0

    # Training parameters
    dropout: float = 0.1
    bias: bool = False
    norm_type: str = "rmsnorm"
    activation: str = "gelu"

    # Optimization parameters
    use_gradient_checkpointing: bool = False
    use_fp8: bool = False
    compile: bool = False

    # RoPE parameters (optional)
    use_rope: bool = True
    rope_theta: float = 10000.0
    rope_scaling: Optional[float] = None

    def __post_init__(self):
        """Validate and compute derived parameters"""
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        self.head_dim = self.n_embd // self.n_head


class AdaptiveMoELLM(nn.Module):
    """Modèle de langage complet avec attention adaptative MoE"""

    def __init__(self, config: AdaptiveMoEConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # Position embeddings or RoPE
        if not config.use_rope:
            self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        else:
            self.position_embedding = None
            # Precompute RoPE frequencies
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis(
                    dim=config.head_dim,
                    end=config.block_size * 2,  # Extra for position interpolation
                    theta=config.rope_theta,
                ),
                persistent=False,
            )

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks avec attention adaptative MoE
        self.layers = nn.ModuleList([
            AdaptiveMoEBlock(
                dim=config.n_embd,
                heads=config.n_head,
                ff_dim=config.n_embd * 4,
                dropout=config.dropout,
                router_temperature=config.router_temperature,
                k_peripheral=config.k_peripheral,
                k_focal=config.k_focal,
                k_reflective=config.k_reflective,
                use_bias=config.bias,
                norm_type=config.norm_type,
                activation=config.activation
            )
            for _ in range(config.n_layer)
        ])

        # Final normalization
        if config.norm_type == "rmsnorm":
            self.ln_final = RMSNorm(config.n_embd)
        else:
            self.ln_final = nn.LayerNorm(config.n_embd)

        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)

        # Weight tying (optional but common)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight') or pn.endswith('fusion.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"AdaptiveMoELLM initialized with {self.get_num_params() / 1e6:.2f}M parameters")

    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.position_embedding is not None:
            n_params -= self.position_embedding.weight.numel()
        return n_params

    def forward(self,
                input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model

        Args:
            input_ids: [batch, seq_len]
            targets: [batch, seq_len] - for loss computation
            attention_mask: [batch, seq_len] - 1 for tokens to attend, 0 for padding

        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: scalar loss if targets provided
        """
        B, N = input_ids.shape
        device = input_ids.device

        # Token embeddings
        x = self.token_embedding(input_ids)  # [B, N, n_embd]

        # Position embeddings
        if self.position_embedding is not None:
            positions = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
            x = x + self.position_embedding(positions)
        # RoPE is applied inside attention if enabled

        x = self.dropout(x)

        # Passage through transformer layers
        routing_stats = []
        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                x, routing_info = torch.utils.checkpoint.checkpoint(
                    layer, x, attention_mask, use_reentrant=False
                )
            else:
                x, routing_info = layer(x, attention_mask)
            routing_stats.append(routing_info)

        # Final norm and output projection
        x = self.ln_final(x)
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Compute cross-entropy loss (like other models in the project)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100  # Standard ignore index for padding tokens
            )

        # Store routing stats for analysis (optional)
        self.last_routing_stats = routing_stats

        return logits, loss

    def get_routing_analysis(self) -> Dict[str, torch.Tensor]:
        """Analyze routing patterns from last forward pass"""
        if not hasattr(self, 'last_routing_stats') or not self.last_routing_stats:
            return {}

        # Average across all layers
        avg_weights = torch.stack([
            stats['weights'] for stats in self.last_routing_stats
        ]).mean(dim=0)

        avg_usage = torch.stack([
            stats['usage'] for stats in self.last_routing_stats
        ]).mean(dim=0)

        return {
            'avg_expert_weights': avg_weights,
            'avg_expert_usage': avg_usage,
            'reflective_weight': avg_weights[0].item(),
            'focal_weight': avg_weights[1].item(),
            'peripheral_weight': avg_weights[2].item(),
        }

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, **kwargs):
        """
        Configure optimizer for training.
        This method is called by the training script.
        """
        # Get optimizer type from kwargs
        optimizer_type = kwargs.get('optimizer_type', 'adamw')

        # Start with all parameters requiring grad
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Create parameter groups with weight decay
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Report number of parameters
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Using {optimizer_type} optimizer")
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create optimizer based on type
        if optimizer_type.lower() == 'lion':
            try:
                from models.optimizers import Lion
                optimizer = Lion(optim_groups, lr=learning_rate, betas=betas)
                print("Using Lion optimizer")
            except ImportError:
                print("Lion optimizer not available, falling back to AdamW")
                optimizer_type = 'adamw'

        if optimizer_type.lower() in ['adamw', 'adam']:
            # Use fused AdamW if available (faster on CUDA)
            use_fused = device_type == 'cuda'

            # Check for foreach parameter (for multi-GPU consistency)
            foreach = not kwargs.get('force_foreach_false', False)

            try:
                optimizer = torch.optim.AdamW(
                    optim_groups,
                    lr=learning_rate,
                    betas=betas,
                    fused=use_fused,
                    foreach=foreach
                )
                print(f"Using {'fused' if use_fused else 'standard'} AdamW")
            except Exception as e:
                # Fallback to standard AdamW if fused fails
                print(f"Fused AdamW failed ({e}), using standard AdamW")
                optimizer = torch.optim.AdamW(
                    optim_groups,
                    lr=learning_rate,
                    betas=betas,
                    foreach=foreach
                )

        return optimizer

    @torch.no_grad()
    def generate(self,
                 idx: Optional[torch.Tensor] = None,
                 max_new_tokens: Optional[int] = None,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 prompt: Optional[torch.Tensor] = None,
                 gen_length: Optional[int] = None) -> Tuple[torch.Tensor, None]:
        """
        Generate text autoregressively

        Args:
            idx: [batch, seq_len] - input token ids (legacy parameter)
            max_new_tokens: number of tokens to generate (legacy)
            temperature: sampling temperature
            top_k: top-k sampling
            top_p: nucleus sampling
            prompt: [batch, seq_len] - input token ids (preferred)
            gen_length: number of tokens to generate (preferred)

        Returns:
            tuple: (generated token ids [batch, seq_len + max_new_tokens], None)
        """
        # Handle compatibility: support both idx=... and prompt=... signatures
        if idx is None and prompt is not None:
            idx = prompt
        elif idx is None and prompt is None:
            raise ValueError("Either idx or prompt must be provided")

        # Handle gen_length vs max_new_tokens
        if gen_length is not None:
            max_new_tokens = gen_length
        elif max_new_tokens is None:
            max_new_tokens = 100  # Default value

        input_ids = idx
        B, N = input_ids.shape
        device = input_ids.device

        for _ in range(max_new_tokens):
            # Crop context if too long
            input_cond = input_ids if input_ids.size(1) <= self.config.block_size else \
                        input_ids[:, -self.config.block_size:]

            # Forward pass
            logits, _ = self(input_cond)

            # Get logits for last position
            logits = logits[:, -1, :] / temperature  # [B, vocab_size]

            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            # Optional nucleus sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        # Return a tuple to match the expected return signature from other models
        return input_ids, None