"""MLA model configuration builders for all MLA variants."""

import torch
from models.models.mla_model import MLAModel, MLAModelConfig
from models.models.mla_selective_model import MLASelectiveModel, MLASelectiveModelConfig
from models.models.parscale_mla import ParScaleMLA, ParScaleMLAConfig, create_parscale_mla
from models.models.mla_llada import MLALLaDAConfig, create_mla_llada_model as _create_mla_llada_model_impl


def create_mla_config(args):
    """Create configuration for MLA-Model.

    Args:
        args: Training arguments containing size, vocab_size, block_size, dropout, bias,
              and optional FP8/optimization parameters

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


def create_mla_model(config):
    """Create MLA-Model instance with the given configuration.

    Args:
        config (MLAModelConfig): Configuration for the MLA model

    Returns:
        MLAModel: Instantiated MLA model
    """
    return MLAModel(config)


def create_mla_selective_config(args):
    """Create configuration for MLA-Selective Model.

    Args:
        args: Training arguments containing size, vocab_size, block_size, dropout, bias,
              and optional FP8/optimization parameters

    Returns:
        MLASelectiveModelConfig: Configuration for MLA-Selective model
    """
    # First get base MLA config
    base_config = create_mla_config(args)

    # Create MLA Selective config with base MLA parameters
    config = MLASelectiveModelConfig(
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
        fp8_params=base_config.fp8_params,
        fp8_mla_params=base_config.fp8_mla_params,
        dropout=base_config.dropout,
        bias=base_config.bias,
        attention_backend=base_config.attention_backend,
        # Disable gradient checkpointing if using torch.compile
        use_gradient_checkpointing=base_config.use_gradient_checkpointing,
    )

    return config


def create_mla_selective_model(config):
    """Create MLA-Selective Model instance with the given configuration.

    Args:
        config (MLASelectiveModelConfig): Configuration for the MLA-Selective model

    Returns:
        MLASelectiveModel: Instantiated MLA-Selective model
    """
    return MLASelectiveModel(config)


def create_parscale_mla_config(args):
    """Create configuration for ParScale-MLA model.

    Args:
        args: Training arguments containing size, vocab_size, block_size, dropout, bias,
              optional FP8/optimization parameters, and ParScale-specific parameters
              (parallel_streams, prefix_length, latent_prefix_length, etc.)

    Returns:
        ParScaleMLAConfig: Configuration for ParScale-MLA model
    """
    # First get base MLA config
    base_config = create_mla_config(args)

    # Create ParScale config with MLA parameters
    config = ParScaleMLAConfig(
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
        use_moe=base_config.use_moe,
        rope_theta=base_config.rope_theta,
        fp8_params=base_config.fp8_params,
        fp8_mla_params=base_config.fp8_mla_params,
        dropout=base_config.dropout,
        bias=base_config.bias,
        attention_backend=base_config.attention_backend,
        use_gradient_checkpointing=base_config.use_gradient_checkpointing,

        # Add ParScale specific parameters
        parallel_streams=getattr(args, 'parallel_streams', 8),
        prefix_length=getattr(args, 'prefix_length', 48),
        latent_prefix_length=getattr(args, 'latent_prefix_length', 16),
        aggregator_epsilon=getattr(args, 'aggregator_epsilon', 0.1),
        diversity_weight=getattr(args, 'diversity_weight', 0.1),
        use_dynamic_inference=getattr(args, 'use_dynamic_inference', True),
        complexity_threshold=getattr(args, 'complexity_threshold', 0.5),
        freeze_base_in_stage2=getattr(args, 'freeze_base_in_stage2', True),
    )

    return config


def create_parscale_mla_model(config, args):
    """Create ParScale-MLA model instance.

    Args:
        config (ParScaleMLAConfig): Configuration for the ParScale-MLA model
        args: Training arguments containing training_stage, base_checkpoint, and size

    Returns:
        ParScaleMLA: Instantiated ParScale-MLA model
    """
    # Check if we're in stage 2 and have a base checkpoint
    if getattr(args, 'training_stage', 1) == 2 and getattr(args, 'base_checkpoint', None):
        print(f"Loading base model from checkpoint: {args.base_checkpoint}")
        # Load base model checkpoint
        checkpoint = torch.load(args.base_checkpoint, map_location='cpu')

        # Extract base model state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint

        # Create base MLA model and load weights
        base_model = MLAModel(config)
        base_model.load_state_dict(state_dict, strict=False)

        # Create ParScale model with pre-trained base
        model = ParScaleMLA(base_model=base_model, config=config)
    else:
        # Create ParScale model from scratch
        model = create_parscale_mla(size=args.size, parallel_streams=config.parallel_streams)

    # Set training stage
    model.set_training_stage(getattr(args, 'training_stage', 1))

    return model


def create_mla_llada_config(args):
    """Create configuration for MLA-LLaDA model.

    Args:
        args: Training arguments containing size, vocab_size, block_size, dropout, bias,
              optional FP8/optimization parameters, and LLaDA-specific parameters
              (mask_ratio_min, mask_ratio_max, remasking_strategy, use_dyt, etc.)

    Returns:
        MLALLaDAConfig: Configuration for MLA-LLaDA model
    """
    # Base dimensions based on size - optimized for target parameter counts
    # Note: With 128k vocab, embeddings+head take ~2*vocab_size*hidden_size parameters
    if args.size == 'small':  # Target: ~500M parameters
        hidden_size = 768  # Increased from 256
        num_layers = 12   # Reduced from 20
        intermediate_size = 1024  # Increased proportionally
        kv_lora_rank = 64  # Increased from 32
    elif args.size == 'medium':  # Target: ~1B parameters
        hidden_size = 384
        num_layers = 24
        intermediate_size = 1536
        kv_lora_rank = 48
    elif args.size == 'large':  # Target: ~2B parameters
        hidden_size = 512
        num_layers = 32
        intermediate_size = 2048
        kv_lora_rank = 64
    else:  # xl - Target: ~4B parameters
        hidden_size = 768
        num_layers = 40
        intermediate_size = 2560
        kv_lora_rank = 96

    config = MLALLaDAConfig(
        # Base GPTConfig parameters
        n_embd=hidden_size,
        n_layer=num_layers,
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        dropout=args.dropout,
        bias=args.bias,

        # MLA-LLaDA specific
        hidden_size=hidden_size,
        num_layers=num_layers,

        # MLA parameters
        q_lora_rank=0,  # Full rank for queries
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,

        # LLaDA parameters
        mask_token_id=args.vocab_size - 1,  # Use last valid token ID
        max_diffusion_steps=50,
        min_diffusion_steps=10,
        mask_ratio_min=getattr(args, 'mask_ratio_min', 0.15),
        mask_ratio_max=getattr(args, 'mask_ratio_max', 0.85),
        remasking_strategy=getattr(args, 'remasking_strategy', 'low_confidence'),

        # FP8 configuration
        use_fp8=getattr(args, 'use_fp8', False),
        fp8_format='e4m3',

        # DynamicTanh
        use_dyt=getattr(args, 'use_dyt', False),
        dyt_alpha_init=getattr(args, 'dyt_alpha_init', 0.5),

        # General parameters
        intermediate_size=intermediate_size,
        gradient_checkpointing=getattr(args, 'gradient_checkpointing', True),
    )

    return config


def create_mla_llada_model(config):
    """Create MLA-LLaDA model instance.

    Args:
        config (MLALLaDAConfig): Configuration for the MLA-LLaDA model

    Returns:
        MLALLaDAModel: Instantiated MLA-LLaDA model
    """
    return _create_mla_llada_model_impl(config)