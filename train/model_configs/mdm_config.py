"""Masked Diffusion Model (MDM) configuration builder."""

from models.config import MDMConfig
from models.mdm.model import MDMModel


def create_mdm_config(args):
    """Create configuration for Masked Diffusion Model based on model size.

    Args:
        args: Training arguments containing size, vocab_size, block_size, dropout, and bias

    Returns:
        MDMConfig: Configuration for MDM model
    """
    # Common configuration parameters
    common_params = {
        'block_size': args.block_size,
        'vocab_size': args.vocab_size,
        'dropout': args.dropout,
        'bias': args.bias,
        'mask_token_id': args.vocab_size - 1,
        'attention_backend': getattr(args, 'attention_backend', None)
    }

    if args.size == 'small':
        config = MDMConfig(
            n_layer=12,
            n_head=12,
            n_embd=768,
            **common_params
        )
    elif args.size == 'medium':
        config = MDMConfig(
            n_layer=24,
            n_head=16,
            n_embd=1024,
            **common_params
        )
    elif args.size == 'large':
        config = MDMConfig(
            n_layer=32,
            n_head=16,
            n_embd=1536,
            **common_params
        )
    else:  # xl
        config = MDMConfig(
            n_layer=40,
            n_head=20,
            n_embd=2560,
            **common_params
        )

    return config


def create_mdm_model(config):
    """Create MDM model instance.

    Args:
        config: MDMConfig instance

    Returns:
        MDMModel: Instantiated MDM model
    """
    return MDMModel(config)