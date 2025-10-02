"""Diffusion-based model configurations (LLaDA and SEDD)."""

from models.llada.model import LLaDAConfig
from models.sedd.model import SEDDConfig


def create_llada_config(args):
    """Create LLaDA configuration based on model size.

    Args:
        args: Training arguments containing size, block_size, vocab_size, dropout, bias,
              and optional BD3 parameters

    Returns:
        LLaDAConfig: Configuration for LLaDA model
    """
    # Common BD3 parameters
    bd3_params = {}
    if hasattr(args, 'bd3_block_length'):
        bd3_params['bd3_block_length'] = args.bd3_block_length
    if hasattr(args, 'bd3_beta'):
        bd3_params['bd3_beta'] = args.bd3_beta
    if hasattr(args, 'bd3_omega'):
        bd3_params['bd3_omega'] = args.bd3_omega
    if hasattr(args, 'disable_entropy_regularization'):
        bd3_params['disable_entropy_regularization'] = args.disable_entropy_regularization

    if args.size == 'small':
        config = LLaDAConfig(
            block_size=args.block_size,
            vocab_size=args.vocab_size,
            n_layer=8,
            n_head=8,
            n_embd=768,
            dropout=args.dropout,
            bias=args.bias,
            ratio_kv=8,
            use_checkpoint=False,
            **bd3_params
        )
    elif args.size == 'medium':
        config = LLaDAConfig(
            block_size=args.block_size,
            vocab_size=args.vocab_size,
            n_layer=16,
            n_head=16,
            n_embd=1024,
            dropout=args.dropout,
            bias=args.bias,
            ratio_kv=8,
            use_checkpoint=False,
            **bd3_params
        )
    else:  # large
        config = LLaDAConfig(
            block_size=args.block_size,
            vocab_size=args.vocab_size,
            n_layer=24,
            n_head=16,
            n_embd=1536,
            dropout=args.dropout,
            bias=args.bias,
            ratio_kv=8,
            use_checkpoint=False,
            **bd3_params
        )
    return config


def create_sedd_config(args):
    """Create SEDD configuration based on model size.

    Args:
        args: Training arguments containing size, block_size, vocab_size, dropout, bias,
              and optional SEDD-specific parameters

    Returns:
        SEDDConfig: Configuration for SEDD model
    """
    if args.size == 'small':
        config = SEDDConfig(
            block_size=args.block_size,
            vocab_size=args.vocab_size,
            n_layer=12,
            n_head=12,
            n_embd=768,
            dropout=args.dropout,
            bias=args.bias,
            mask_token_id=args.vocab_size - 1,
            cond_dim=128,
            scale_by_sigma=True,
            mlp_ratio=4,
            graph_type=getattr(args, 'graph_type', 'absorb'),
            noise_type=getattr(args, 'noise_type', 'loglinear'),
            sigma_min=getattr(args, 'sigma_min', 1e-4),
            sigma_max=getattr(args, 'sigma_max', 20.0),
            use_gradient_checkpointing=False,
            attention_backend=getattr(args, 'attention_backend', None),
            use_fp8=getattr(args, 'use_fp8', False),
            use_dyt=getattr(args, 'use_dyt', False),
            dyt_alpha_init=getattr(args, 'dyt_alpha_init', 0.5),
        )
    elif args.size == 'medium':
        config = SEDDConfig(
            block_size=args.block_size,
            vocab_size=args.vocab_size,
            n_layer=16,
            n_head=16,
            n_embd=1024,
            dropout=args.dropout,
            bias=args.bias,
            mask_token_id=args.vocab_size - 1,
            cond_dim=128,
            scale_by_sigma=True,
            mlp_ratio=4,
            graph_type=getattr(args, 'graph_type', 'absorb'),
            noise_type=getattr(args, 'noise_type', 'loglinear'),
            sigma_min=getattr(args, 'sigma_min', 1e-4),
            sigma_max=getattr(args, 'sigma_max', 20.0),
            use_gradient_checkpointing=False,
            attention_backend=getattr(args, 'attention_backend', None),
            use_fp8=getattr(args, 'use_fp8', False),
            use_dyt=getattr(args, 'use_dyt', False),
            dyt_alpha_init=getattr(args, 'dyt_alpha_init', 0.5),
        )
    else:  # large
        config = SEDDConfig(
            block_size=args.block_size,
            vocab_size=args.vocab_size,
            n_layer=24,
            n_head=16,
            n_embd=1536,
            dropout=args.dropout,
            bias=args.bias,
            mask_token_id=args.vocab_size - 1,
            cond_dim=128,
            scale_by_sigma=True,
            mlp_ratio=4,
            graph_type=getattr(args, 'graph_type', 'absorb'),
            noise_type=getattr(args, 'noise_type', 'loglinear'),
            sigma_min=getattr(args, 'sigma_min', 1e-4),
            sigma_max=getattr(args, 'sigma_max', 20.0),
            use_gradient_checkpointing=False,
            attention_backend=getattr(args, 'attention_backend', None),
            use_fp8=getattr(args, 'use_fp8', False),
            use_dyt=getattr(args, 'use_dyt', False),
            dyt_alpha_init=getattr(args, 'dyt_alpha_init', 0.5),
        )
    return config