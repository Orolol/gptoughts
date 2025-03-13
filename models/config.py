"""Configuration classes for various model architectures."""

from dataclasses import dataclass
from typing import Optional, Literal

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
    use_gradient_checkpointing: bool = False  # Whether to use gradient checkpointing to save memory

@dataclass
class LLaDAConfig:
    """Configuration class for LLaDA model hyperparameters."""
    
    block_size: int = 1024  # Maximum sequence length
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64
    n_layer: int = 12       # Number of transformer layers
    n_head: int = 12        # Number of attention heads
    n_embd: int = 768      # Embedding dimension
    dropout: float = 0.0    # Dropout probability
    bias: bool = True       # Whether to use bias in Linear and LayerNorm layers
    ratio_kv: int = 4      # Ratio of key/value heads
    mask_token_id: int = 126336  # Special token ID for [MASK]
    num_experts: int = 8    # Number of experts in the MoE layer
    k: int = 2             # Top-k experts to route tokens to
    temperature: float = 0.0  # Temperature for sampling during generation
    remasking: str = 'low_confidence'  # Remasking strategy: 'low_confidence' or 'random'
    use_checkpoint: bool = False  # Whether to use gradient checkpointing

@dataclass
class ModelArgs:
    """
    Data class for DeepSeek model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1. 