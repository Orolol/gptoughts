"""
Configuration for Adaptive MoE Model
"""

from models.models.adaptive_moe_model import AdaptiveMoEConfig, AdaptiveMoELLM


def create_adaptive_moe_config(args):
    """Create AdaptiveMoE configuration based on command line arguments"""

    # Size-based configurations
    if args.size == "small":
        n_layer = 12
        n_embd = 768
        n_head = 12
        k_peripheral = 32
        k_focal = 64
        k_reflective = 128
    elif args.size == "medium":
        n_layer = 16
        n_embd = 1024
        n_head = 16
        k_peripheral = 48
        k_focal = 96
        k_reflective = 192
    elif args.size == "large":
        n_layer = 32
        n_embd = 2048
        n_head = 32
        k_peripheral = 64
        k_focal = 128
        k_reflective = 256
    elif args.size == "xl":
        n_layer = 40
        n_embd = 2560
        n_head = 20
        k_peripheral = 80
        k_focal = 160
        k_reflective = 320
    else:
        raise ValueError(f"Unknown model size: {args.size}")

    # Get optional parameters with defaults
    router_temperature = getattr(args, 'router_temperature', 1.0)
    use_rope = getattr(args, 'use_rope', True)
    rope_theta = getattr(args, 'rope_theta', 10000.0)
    rope_scaling = getattr(args, 'rope_scaling', None)
    norm_type = getattr(args, 'norm_type', 'rmsnorm')
    activation = getattr(args, 'activation', 'gelu')

    # Override k values if provided
    if hasattr(args, 'k_peripheral'):
        k_peripheral = args.k_peripheral
    if hasattr(args, 'k_focal'):
        k_focal = args.k_focal
    if hasattr(args, 'k_reflective'):
        k_reflective = args.k_reflective

    config = AdaptiveMoEConfig(
        vocab_size=args.vocab_size,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        block_size=args.block_size,
        # MoE attention parameters
        k_peripheral=k_peripheral,
        k_focal=k_focal,
        k_reflective=k_reflective,
        router_temperature=router_temperature,
        # Training parameters
        dropout=args.dropout,
        bias=args.bias,
        norm_type=norm_type,
        activation=activation,
        # Optimization parameters
        use_gradient_checkpointing=getattr(args, 'use_gradient_checkpointing', False),
        use_fp8=getattr(args, 'use_fp8', False),
        compile=getattr(args, 'compile', False),
        # RoPE parameters
        use_rope=use_rope,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
    )

    return config


def create_adaptive_moe_model(config):
    """Create AdaptiveMoELLM model from configuration"""
    return AdaptiveMoELLM(config)