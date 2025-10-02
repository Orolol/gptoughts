"""
SLM-MoE-MLA Model: Small Language Model with Mixture of Experts and Multi-head Latent Attention.

This model is designed for ~200M parameters and combines:
- Multi-head Latent Attention (MLA) for efficient attention computation
- Mixture of Experts (MoE) with ultra-high shared weights (90% shared, 10% specialized)
- Dynamic Tanh (DyT) normalization for faster training
- RoPE positional encoding
- Optimized for hobby-scale training with maximum efficiency
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional, Dict, List, Tuple, Union, Any
from dataclasses import dataclass

# Import MLA components
from models.blocks.mla import MLA
from models.blocks.mla_fp8 import MLA_FP8
from models.blocks.normalization import RMSNorm, DynamicTanh
from models.blocks.positional_encoding import RoPE
from models.blocks.tensor_utils import isolate_tensor

# Import utility functions
from train.train_utils import estimate_mfu as utils_estimate_mfu

@dataclass
class SLMConfig:
    """Configuration for SLM-MoE-MLA Model optimized for ~200M parameters."""
    # Architecture - optimized for 200M params
    n_layer: int = 10
    n_embd: int = 768
    n_head: int = 12
    n_inner: Optional[int] = None  # Inner dimension for MLP. If None, will be 3*n_embd for SLM
    vocab_size: int = 50304
    block_size: int = 2048
    
    # MLA specific parameters - optimized for SLM
    q_lora_rank: int = 0
    kv_lora_rank: int = 256  # Reduced from 512 for SLM
    qk_nope_head_dim: int = 96  # Reduced for SLM
    qk_rope_head_dim: int = 32  # Reduced for SLM
    v_head_dim: int = 96  # Reduced for SLM
    
    # MoE parameters - ultra-high sharing
    num_experts: int = 32  # Large number of experts
    experts_per_token: int = 4  # Top-k routing
    shared_weight_ratio: float = 0.90  # 90% of weights are shared between experts
    
    # Router parameters
    router_temperature: float = 0.1
    router_z_loss_coef: float = 0.001
    load_balance_coef: float = 0.01
    
    # Attention backend
    attention_backend: Optional[str] = None
    attn_impl: str = "absorb"  # "absorb" (optimized) or "naive" (standard)
    
    # RoPE
    original_max_seq_len: int = 2048
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: float = 10000.0
    
    # Precision
    use_fp8: bool = False  # Master switch for FP8 usage
    fp8_tile_size: int = 128  # Tile size for FP8 quantization
    
    # Dropout and regularization
    dropout: float = 0.0
    bias: bool = False
    
    # Training
    use_gradient_checkpointing: bool = True
    init_device: str = 'cuda'
    label_smoothing: float = 0.0
    
    # Dynamic Tanh (DyT) options - enabled by default for SLM
    use_dyt: bool = True
    dyt_alpha_init: float = 0.5
    
    def __post_init__(self):
        # Set inner dimension if not provided - smaller for SLM
        if self.n_inner is None:
            self.n_inner = 3 * self.n_embd  # 3x instead of 4x for efficiency


class SLMRouter(nn.Module):
    """
    Optimized router for SLM with many experts and ultra-high weight sharing.
    Includes improved load balancing and temperature scaling.
    """
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        
        # Router projection with better initialization for many experts
        self.router_proj = nn.Linear(config.n_embd, config.num_experts, bias=False)
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.tensor(config.router_temperature))
        
        # Load balancing loss coefficients
        self.router_z_loss_coef = config.router_z_loss_coef
        self.load_balance_coef = config.load_balance_coef
        
        # Initialize router weights with smaller variance for stability
        nn.init.normal_(self.router_proj.weight, mean=0.0, std=0.01)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.
        
        Args:
            x: Input tensor [batch_size, seq_len, n_embd]
            
        Returns:
            expert_indices: Selected expert indices [batch_size, seq_len, experts_per_token]
            expert_weights: Normalized weights for selected experts [batch_size, seq_len, experts_per_token]
            router_loss: Combined router loss (z-loss + load balancing)
        """
        batch_size, seq_len, n_embd = x.shape
        
        # Compute router logits
        router_logits = self.router_proj(x)  # [batch_size, seq_len, num_experts]
        
        # Apply temperature scaling
        router_logits = router_logits / (self.temperature.abs() + 1e-8)
        
        # Compute routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        expert_weights, expert_indices = torch.topk(
            router_probs, self.experts_per_token, dim=-1
        )  # [batch_size, seq_len, experts_per_token]
        
        # Normalize weights
        expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Compute auxiliary losses
        router_loss = self._compute_router_losses(router_logits, router_probs)
        
        return expert_indices, expert_weights, router_loss
    
    def _compute_router_losses(self, router_logits: torch.Tensor, 
                              router_probs: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary losses for router training."""
        # Router z-loss (encourages router logits to stay small)
        z_loss = torch.square(router_logits).mean()
        
        # Load balancing loss (encourages uniform distribution across experts)
        # Compute actual load per expert
        tokens_per_expert = router_probs.mean(dim=(0, 1))  # [num_experts]
        
        # Ideal load would be uniform
        ideal_load = torch.ones_like(tokens_per_expert) / self.num_experts
        
        # KL divergence between actual and ideal load
        load_balance_loss = F.kl_div(
            tokens_per_expert.log(),
            ideal_load,
            reduction='batchmean',
            log_target=False
        )
        
        # Combine losses
        total_loss = (
            self.router_z_loss_coef * z_loss +
            self.load_balance_coef * load_balance_loss
        )
        
        return total_loss


class SLMSharedExpertMLP(nn.Module):
    """
    Ultra-efficient MLP with 90% shared weights for SLM.
    Only 10% of parameters are expert-specific.
    """
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.shared_ratio = config.shared_weight_ratio
        
        # Calculate dimensions
        self.hidden_dim = config.n_inner
        self.shared_dim = int(self.hidden_dim * self.shared_ratio)
        self.expert_dim = self.hidden_dim - self.shared_dim
        
        # Shared components (90% of parameters)
        self.shared_up_proj = nn.Linear(config.n_embd, self.shared_dim, bias=False)
        self.shared_down_proj = nn.Linear(self.shared_dim, config.n_embd, bias=False)
        
        # Expert-specific components (10% of parameters per expert)
        # Use separate layers for each expert to maximize differentiation
        self.expert_up_proj = nn.ModuleList([
            nn.Linear(config.n_embd, self.expert_dim, bias=False)
            for _ in range(config.num_experts)
        ])
        self.expert_down_proj = nn.ModuleList([
            nn.Linear(self.expert_dim, config.n_embd, bias=False)
            for _ in range(config.num_experts)
        ])
        
        # Mixing weights to combine shared and expert outputs
        self.mix_weights = nn.Parameter(
            torch.ones(config.num_experts, 2) * 0.5  # [num_experts, 2] for [shared, expert]
        )
        
        # Activation function
        self.activation = nn.SiLU()
        
        # Initialize weights
        self._init_weights()
    
    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor, 
                expert_weights: torch.Tensor) -> torch.Tensor:
        """
        Fully vectorized forward pass (torch.compile compatible).
        
        Args:
            x: Input tensor [batch_size, seq_len, n_embd]
            expert_indices: Selected expert indices [batch_size, seq_len, experts_per_token]
            expert_weights: Weights for selected experts [batch_size, seq_len, experts_per_token]
        """
        batch_size, seq_len, n_embd = x.shape
        experts_per_token = expert_indices.shape[-1]
        
        # Shared processing (applied to all tokens)
        shared_hidden = self.activation(self.shared_up_proj(x))
        shared_output = self.shared_down_proj(shared_hidden)
        
        # Flatten for batch processing
        x_flat = x.view(-1, n_embd)  # [B*T, n_embd]
        expert_indices_flat = expert_indices.view(-1, experts_per_token)  # [B*T, experts_per_token]
        expert_weights_flat = expert_weights.view(-1, experts_per_token)  # [B*T, experts_per_token]
        shared_output_flat = shared_output.view(-1, n_embd)  # [B*T, n_embd]
        
        num_tokens = x_flat.shape[0]
        
        # Vectorized expert processing
        # Expand inputs for all expert selections: [B*T, experts_per_token] -> [B*T*experts_per_token]
        x_expanded = x_flat.unsqueeze(1).expand(-1, experts_per_token, -1).reshape(-1, n_embd)  # [B*T*K, n_embd]
        shared_expanded = shared_output_flat.unsqueeze(1).expand(-1, experts_per_token, -1).reshape(-1, n_embd)  # [B*T*K, n_embd]
        indices_flat = expert_indices_flat.reshape(-1)  # [B*T*K]
        weights_flat = expert_weights_flat.reshape(-1)  # [B*T*K]
        
        # Stack expert parameters for batch indexing (no bias since bias=False)
        expert_up_weights = torch.stack([proj.weight for proj in self.expert_up_proj])  # [num_experts, expert_dim, n_embd]
        expert_down_weights = torch.stack([proj.weight for proj in self.expert_down_proj])  # [num_experts, n_embd, expert_dim]
        
        # Select parameters for each token-expert pair
        selected_up_w = expert_up_weights[indices_flat]  # [B*T*K, expert_dim, n_embd]
        selected_down_w = expert_down_weights[indices_flat]  # [B*T*K, n_embd, expert_dim]
        
        # Batch matrix multiplication for expert up projection
        x_for_bmm = x_expanded.unsqueeze(-1)  # [B*T*K, n_embd, 1]
        expert_hidden = torch.bmm(selected_up_w, x_for_bmm).squeeze(-1)  # [B*T*K, expert_dim]
        expert_hidden = self.activation(expert_hidden)
        
        # Batch matrix multiplication for expert down projection
        expert_hidden_for_bmm = expert_hidden.unsqueeze(-1)  # [B*T*K, expert_dim, 1]
        expert_specific_out = torch.bmm(selected_down_w, expert_hidden_for_bmm).squeeze(-1)  # [B*T*K, n_embd]
        
        # Get mixing weights
        mix_weights = torch.sigmoid(self.mix_weights)  # [num_experts, 2]
        selected_mix_weights = mix_weights[indices_flat]  # [B*T*K, 2]
        
        mix_shared = selected_mix_weights[:, 0]
        mix_expert = selected_mix_weights[:, 1]
        
        # Normalize mixing weights
        total_mix = mix_shared + mix_expert + 1e-8
        mix_shared = mix_shared / total_mix
        mix_expert = mix_expert / total_mix
        
        # Mix shared and expert outputs
        mixed_output = (mix_shared.unsqueeze(-1) * shared_expanded + 
                       mix_expert.unsqueeze(-1) * expert_specific_out)  # [B*T*K, n_embd]
        
        # Apply routing weights
        weighted_output = weights_flat.unsqueeze(-1) * mixed_output  # [B*T*K, n_embd]
        
        # Reshape back to [B*T, experts_per_token, n_embd] and sum across experts
        final_output = weighted_output.view(num_tokens, experts_per_token, n_embd).sum(dim=1)  # [B*T, n_embd]
        
        # Reshape to original dimensions
        return final_output.view(batch_size, seq_len, n_embd)
    
    def _init_weights(self):
        """Initialize weights with scaled initialization."""
        scale = 1.0 / math.sqrt(self.config.n_embd)
        
        # Shared weights (larger scale since they're used more)
        nn.init.normal_(self.shared_up_proj.weight, mean=0.0, std=scale)
        nn.init.normal_(self.shared_down_proj.weight, mean=0.0, std=scale)
        
        # Expert weights (smaller initialization for stability)
        expert_scale = scale * 0.05  # Very small for expert-specific parts
        for expert_id in range(self.num_experts):
            nn.init.normal_(self.expert_up_proj[expert_id].weight, mean=0.0, std=expert_scale)
            nn.init.normal_(self.expert_down_proj[expert_id].weight, mean=0.0, std=expert_scale)
        
        # Initialize mixing weights to favor shared processing initially
        with torch.no_grad():
            self.mix_weights[:, 0] = 2.0  # Higher bias toward shared
            self.mix_weights[:, 1] = -1.0  # Lower bias toward expert-specific


class SLMBlock(nn.Module):
    """
    SLM Transformer block with MLA attention and MoE MLP.
    """
    def __init__(self, config: SLMConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Choose normalization
        if config.use_dyt:
            self.ln_1 = DynamicTanh(config.n_embd, config.dyt_alpha_init)
            self.ln_2 = DynamicTanh(config.n_embd, config.dyt_alpha_init)
        else:
            self.ln_1 = RMSNorm(config.n_embd)
            self.ln_2 = RMSNorm(config.n_embd)
        
        # MLA attention
        if config.use_fp8:
            self.attn = MLA_FP8(config)
        else:
            self.attn = MLA(config)
        
        # MoE MLP
        self.router = SLMRouter(config)
        self.mlp = SLMSharedExpertMLP(config)
        
    def forward(self, x: torch.Tensor, past_key_value=None, 
                use_cache=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        # Attention block
        residual = x
        x = self.ln_1(x)
        
        if use_cache and past_key_value is not None:
            attn_output, present = self.attn(x, start_pos=0, past_key_value=past_key_value, use_cache=use_cache)
        else:
            attn_output = self.attn(x, start_pos=0)
            present = None
        
        x = residual + attn_output
        
        # MoE block
        residual = x
        x = self.ln_2(x)
        
        # Route tokens to experts
        expert_indices, expert_weights, router_loss = self.router(x)
        
        # Process through MoE MLP
        mlp_output = self.mlp(x, expert_indices, expert_weights)
        
        x = residual + mlp_output
        
        return x, present, router_loss


class SLMMLA(nn.Module):
    """
    SLM-MoE-MLA: Small Language Model with Mixture of Experts and Multi-head Latent Attention.
    Optimized for ~200M parameters with ultra-high weight sharing.
    """
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings (shared across all experts)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            SLMBlock(config, i) for i in range(config.n_layer)
        ])
        
        # Final layer norm
        if config.use_dyt:
            self.ln_f = DynamicTanh(config.n_embd, config.dyt_alpha_init)
        else:
            self.ln_f = RMSNorm(config.n_embd)
        
        # Language modeling head (shared across all experts)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie embeddings and output weights
        self.lm_head.weight = self.wte.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report parameter count
        self.report_parameter_count()
        
    def _init_weights(self, module):
        """Initialize weights using GPT-style initialization."""
        if isinstance(module, nn.Linear):
            # Special handling for specific layers
            if hasattr(module, '_is_expert_layer'):
                # Expert layers get smaller initialization
                std = 0.02 / math.sqrt(2 * self.config.n_layer) * 0.1
            else:
                std = 0.02
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def report_parameter_count(self):
        """Report detailed parameter count breakdown."""
        total_params = sum(p.numel() for p in self.parameters())
        
        # Count different components
        embed_params = self.wte.weight.numel()
        
        # Count MLA parameters
        mla_params = 0
        for block in self.blocks:
            mla_params += sum(p.numel() for p in block.attn.parameters())
        
        # Count shared MLP parameters
        shared_mlp_params = 0
        expert_mlp_params = 0
        router_params = 0
        
        for block in self.blocks:
            # Shared MLP
            shared_mlp_params += block.mlp.shared_up_proj.weight.numel()
            shared_mlp_params += block.mlp.shared_down_proj.weight.numel()
            shared_mlp_params += block.mlp.mix_weights.numel()
            
            # Expert MLP
            for expert_id in range(self.config.num_experts):
                expert_mlp_params += block.mlp.expert_up_proj[expert_id].weight.numel()
                expert_mlp_params += block.mlp.expert_down_proj[expert_id].weight.numel()
            
            # Router
            router_params += sum(p.numel() for p in block.router.parameters())
        
        # Normalization parameters
        norm_params = 0
        for block in self.blocks:
            norm_params += sum(p.numel() for p in block.ln_1.parameters())
            norm_params += sum(p.numel() for p in block.ln_2.parameters())
        norm_params += sum(p.numel() for p in self.ln_f.parameters())
        
        print(f"\n{'='*60}")
        print(f"SLM-MoE-MLA Parameter Breakdown:")
        print(f"  Total parameters: {total_params/1e6:.2f}M")
        print(f"  - Embeddings: {embed_params/1e6:.2f}M ({embed_params/total_params*100:.1f}%)")
        print(f"  - MLA Attention: {mla_params/1e6:.2f}M ({mla_params/total_params*100:.1f}%)")
        print(f"  - Shared MLP: {shared_mlp_params/1e6:.2f}M ({shared_mlp_params/total_params*100:.1f}%)")
        print(f"  - Expert MLP: {expert_mlp_params/1e6:.2f}M ({expert_mlp_params/total_params*100:.1f}%)")
        print(f"  - Routers: {router_params/1e6:.2f}M ({router_params/total_params*100:.1f}%)")
        print(f"  - Normalization: {norm_params/1e6:.2f}M ({norm_params/total_params*100:.1f}%)")
        print(f"\nMoE Configuration:")
        print(f"  - Number of experts: {self.config.num_experts}")
        print(f"  - Experts per token: {self.config.experts_per_token}")
        print(f"  - Shared weight ratio: {self.config.shared_weight_ratio*100:.1f}%")
        print(f"  - Expert parameters per layer: {expert_mlp_params/self.config.n_layer/1e6:.2f}M")
        print(f"  - Effective parameters: {(total_params - expert_mlp_params + expert_mlp_params/self.config.num_experts*self.config.experts_per_token)/1e6:.2f}M")
        print(f"{'='*60}\n")
    
    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None,
                past_key_values=None, use_cache=False) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.wte(input_ids)
        
        # Store router losses
        router_losses = []
        
        # Forward through transformer blocks
        presents = []
        for i, block in enumerate(self.blocks):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            if self.config.use_gradient_checkpointing and self.training:
                x, present, router_loss = checkpoint.checkpoint(
                    block, x, past_key_value, use_cache, use_reentrant=False
                )
            else:
                x, present, router_loss = block(x, past_key_value, use_cache)
            
            router_losses.append(router_loss)
            if use_cache:
                presents.append(present)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            # Shift labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        # Compute total router loss
        total_router_loss = torch.stack(router_losses).sum() if router_losses else torch.tensor(0.0, device=logits.device)
        
        outputs = {
            'logits': logits,
            'loss': loss,
            'router_loss': total_router_loss
        }
        
        if use_cache:
            outputs['past_key_values'] = presents
            
        return outputs


def create_slm_model(size: str = 'small', **kwargs) -> SLMMLA:
    """
    Create SLM model with different size configurations.
    
    Args:
        size: Model size ('tiny', 'small', 'medium')
        **kwargs: Additional configuration overrides
    """
    if size == 'tiny':
        # ~100M parameters
        config = SLMConfig(
            n_layer=8,
            n_embd=512,
            n_head=8,
            num_experts=16,
            experts_per_token=2,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
            **kwargs
        )
    elif size == 'small':
        # ~200M parameters
        config = SLMConfig(
            n_layer=10,
            n_embd=768,
            n_head=12,
            num_experts=32,
            experts_per_token=4,
            kv_lora_rank=256,
            qk_nope_head_dim=96,
            qk_rope_head_dim=32,
            v_head_dim=96,
            **kwargs
        )
    elif size == 'medium':
        # ~400M parameters
        config = SLMConfig(
            n_layer=12,
            n_embd=1024,
            n_head=16,
            num_experts=48,
            experts_per_token=6,
            kv_lora_rank=384,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown size: {size}. Supported sizes: tiny, small, medium")
    
    return SLMMLA(config)