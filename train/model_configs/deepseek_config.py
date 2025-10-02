"""DeepSeek model configuration builder."""

from models.deepseek.deepseek_adapter_mtp import DeepSeekMiniConfigMTP


def create_deepseek_config(args):
    """Create DeepSeek configuration based on model size.

    Args:
        args: Training arguments containing size, vocab_size, block_size, dropout, and bias

    Returns:
        DeepSeekMiniConfigMTP: Configuration for DeepSeek model
    """
    if args.size == 'small':
        config = DeepSeekMiniConfigMTP(
            vocab_size=args.vocab_size,
            hidden_size=1024,
            num_hidden_layers=8,
            num_attention_heads=8,
            head_dim=128,
            intermediate_size=2816,
            num_experts=4,
            num_experts_per_token=1,
            max_position_embeddings=max(16, args.block_size),
            kv_compression_dim=64,
            query_compression_dim=192,
            rope_head_dim=32,
            dropout=args.dropout,
            attention_dropout=args.dropout,
            hidden_dropout=args.dropout,
            bias=args.bias
        )
    elif args.size == 'medium':
        config = DeepSeekMiniConfigMTP(
            vocab_size=args.vocab_size,
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            head_dim=128,
            intermediate_size=4096,
            num_experts=32,
            num_experts_per_token=4,
            max_position_embeddings=args.block_size,
            kv_compression_dim=128,
            query_compression_dim=384,
            rope_head_dim=32,
            dropout=args.dropout,
            attention_dropout=args.dropout,
            hidden_dropout=args.dropout,
            bias=args.bias
        )
    else:  # large
        config = DeepSeekMiniConfigMTP(
            vocab_size=args.vocab_size,
            hidden_size=3072,
            num_hidden_layers=32,
            num_attention_heads=24,
            head_dim=128,
            intermediate_size=8192,
            num_experts=64,
            num_experts_per_token=4,
            max_position_embeddings=args.block_size,
            kv_compression_dim=256,
            query_compression_dim=768,
            rope_head_dim=32,
            dropout=args.dropout,
            attention_dropout=args.dropout,
            hidden_dropout=args.dropout,
            bias=args.bias
        )
    return config