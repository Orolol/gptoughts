"""MOE-MLA model configuration and creation."""

from models.models.mla_model import MLAModelConfig
from models.models.moe_mla_model import MOEMLA, MOEMLAConfig


def create_mla_config(args):
    """Create configuration for MLA model based on size.

    Args:
        args: Training arguments containing size, vocab_size, block_size, dropout, and bias

    Returns:
        MLAModelConfig: Configuration for MLA model
    """
    # Define key parameters based on size
    if args.size == 'small':
        n_layer = 12
        n_embd = 768
        n_head = 16
        q_lora_rank = 0
        kv_lora_rank = 256
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
    elif args.size == 'medium':
        n_layer = 16
        n_embd = 1024
        n_head = 16
        q_lora_rank = 0
        kv_lora_rank = 512
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
    elif args.size == 'large':
        n_layer = 32
        n_embd = 2048
        n_head = 32
        q_lora_rank = 0
        kv_lora_rank = 512
        qk_nope_head_dim = 192
        qk_rope_head_dim = 96
        v_head_dim = 192
    else:  # xl
        n_layer = 40
        n_embd = 2560
        n_head = 20
        q_lora_rank = 0
        kv_lora_rank = 1024
        qk_nope_head_dim = 384
        qk_rope_head_dim = 192
        v_head_dim = 384

    # Create config object
    config = MLAModelConfig(
        # Architecture
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        vocab_size=args.vocab_size,
        block_size=args.block_size,

        # MLA parameters
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,

        # MoE parameters
        use_moe=False,  # Set to False for dense model

        # RoPE parameters
        rope_theta=10000.0,

        # Precision
        fp8_params=getattr(args, 'use_fp8', False),
        fp8_mla_params=getattr(args, 'fp8_mla_params', False),
        use_fp8=getattr(args, 'use_fp8', False),
        fp8_tile_size=getattr(args, 'fp8_tile_size', 128),

        # Other parameters
        dropout=args.dropout,
        bias=args.bias,
        attention_backend=getattr(args, 'attention_backend', None),
        use_gradient_checkpointing=True,
    )

    return config


def create_moe_mla_config(args):
    """Create configuration for MOE-MLA model.

    Args:
        args: Training arguments containing model configuration parameters

    Returns:
        MOEMLAConfig: Configuration for MOE-MLA model
    """
    # Start with base MLA config dimensions
    base_config = create_mla_config(args)

    # Create MOE-MLA config
    config = MOEMLAConfig(
        # Copy base MLA parameters
        n_layer=base_config.n_layer,
        n_embd=base_config.n_embd,
        n_head=base_config.n_head,
        vocab_size=base_config.vocab_size,
        block_size=base_config.block_size,
        q_lora_rank=base_config.q_lora_rank,
        kv_lora_rank=base_config.kv_lora_rank,
        qk_nope_head_dim=base_config.qk_nope_head_dim,
        qk_rope_head_dim=base_config.qk_rope_head_dim,
        v_head_dim=base_config.v_head_dim,
        rope_theta=base_config.rope_theta,
        dropout=base_config.dropout,
        bias=base_config.bias,
        attention_backend=base_config.attention_backend,
        use_gradient_checkpointing=base_config.use_gradient_checkpointing,

        # MOE-specific parameters
        num_experts=getattr(args, 'num_experts', 16),
        experts_per_token=getattr(args, 'experts_per_token', 2),
        shared_weight_ratio=getattr(args, 'shared_weight_ratio', 0.9),

        # FP8 settings
        use_fp8=getattr(args, 'use_fp8', False),
        fp8_tile_size=getattr(args, 'fp8_tile_size', 128),

        # DyT settings
        use_dyt=getattr(args, 'use_dyt', False),
        dyt_alpha_init=getattr(args, 'dyt_alpha_init', 0.5),
    )

    return config


def create_moe_mla_model(config):
    """Create MOE-MLA model instance.

    Args:
        config: MOEMLAConfig instance

    Returns:
        MOEMLA: MOE-MLA model instance
    """
    return MOEMLA(config)