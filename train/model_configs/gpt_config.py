"""GPT model configuration builder."""

from models.models.model import GPTConfig


def create_gpt_config(args):
    """Create GPT configuration based on model size.

    Args:
        args: Training arguments containing size, vocab_size, block_size, dropout, and bias

    Returns:
        GPTConfig: Configuration for GPT model
    """
    if args.size == 'small':
        config = GPTConfig(
            n_layer=8,
            n_head=8,
            n_embd=768,
            block_size=args.block_size,
            bias=args.bias,
            vocab_size=args.vocab_size,
            dropout=args.dropout,
            attention_backend=getattr(args, 'attention_backend', None)
        )
    elif args.size == 'medium':
        config = GPTConfig(
            n_layer=12,
            n_head=12,
            n_embd=1024,
            block_size=args.block_size,
            bias=args.bias,
            vocab_size=args.vocab_size,
            dropout=args.dropout,
            attention_backend=getattr(args, 'attention_backend', None)
        )
    else:  # large
        config = GPTConfig(
            n_layer=24,
            n_head=16,
            n_embd=1536,
            block_size=args.block_size,
            bias=args.bias,
            vocab_size=args.vocab_size,
            dropout=args.dropout,
            attention_backend=getattr(args, 'attention_backend', None)
        )
    return config