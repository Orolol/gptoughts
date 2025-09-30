"""
Specialized model configurations for SLM, NSA, HSE, and HRM models.

This module contains configuration creation and model instantiation functions
for specialized research models in the GPToughts framework.
"""

from models.models.slm_moe_mla import SLMConfig, SLMMLA
from models.models.nsa_model import NSAModel, NSAModelConfig
from models.models.hse_model import HSEModel, HSEConfig
from models.models.hrm_model import HRM, HRMConfig


def create_slm_config(args):
    """
    Create configuration for SLM-MoE-MLA Model.

    SLM (Sparse Language Model) combines Multi-head Latent Attention (MLA)
    with Mixture of Experts (MoE) for efficient parameter scaling.

    Args:
        args: Training arguments containing model hyperparameters

    Returns:
        SLMConfig: Configuration object for SLM model
    """
    # Define key parameters based on size - optimized for different parameter counts
    if args.size == 'tiny':
        # ~100M parameters
        n_layer = 8
        n_embd = 512
        n_head = 8
        num_experts = 16
        experts_per_token = 2
        kv_lora_rank = 128
        qk_nope_head_dim = 64
        qk_rope_head_dim = 32
        v_head_dim = 64
    elif args.size == 'small':
        # ~200M parameters
        n_layer = 8
        n_embd = 768
        n_head = 12
        num_experts = 32
        experts_per_token = 1
        kv_lora_rank = 256
        qk_nope_head_dim = 96
        qk_rope_head_dim = 32
        v_head_dim = 96
    elif args.size == 'medium':
        # ~400M parameters
        n_layer = 12
        n_embd = 1024
        n_head = 16
        num_experts = 48
        experts_per_token = 6
        kv_lora_rank = 384
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
    else:  # large - ~800M parameters
        n_layer = 16
        n_embd = 1280
        n_head = 20
        num_experts = 64
        experts_per_token = 8
        kv_lora_rank = 512
        qk_nope_head_dim = 160
        qk_rope_head_dim = 80
        v_head_dim = 160

    # Create SLM config
    config = SLMConfig(
        # Architecture
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        vocab_size=args.vocab_size,
        block_size=args.block_size,

        # MLA parameters
        q_lora_rank=0,  # No low-rank for queries in SLM
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,

        # MoE parameters - ultra high sharing
        num_experts=getattr(args, 'num_experts', num_experts),
        experts_per_token=getattr(args, 'experts_per_token', experts_per_token),
        shared_weight_ratio=getattr(args, 'shared_weight_ratio', 0.90),

        # Router parameters
        router_temperature=getattr(args, 'router_temperature', 0.1),
        router_z_loss_coef=getattr(args, 'router_z_loss_coef', 0.001),
        load_balance_coef=getattr(args, 'load_balance_coef', 0.01),

        # RoPE parameters
        rope_theta=10000.0,
        original_max_seq_len=args.block_size,

        # Precision and optimization
        use_fp8=getattr(args, 'use_fp8', False),
        fp8_tile_size=getattr(args, 'fp8_tile_size', 128),

        # Training parameters
        dropout=args.dropout,
        bias=args.bias,
        use_gradient_checkpointing=getattr(args, 'gradient_checkpointing', True),

        # Dynamic Tanh - enabled by default for SLM
        use_dyt=getattr(args, 'use_dyt', True),
        dyt_alpha_init=getattr(args, 'dyt_alpha_init', 0.5),

        # Attention backend
        attention_backend=getattr(args, 'attention_backend', None),
    )

    return config


def create_slm_model(config):
    """
    Create SLM-MoE-MLA model instance.

    Args:
        config: SLMConfig object with model configuration

    Returns:
        SLMMLA: Instantiated SLM model
    """
    return SLMMLA(config)


def create_nsa_config(args):
    """
    Create configuration for NSA Model.

    NSA (Neural Selective Attention) implements efficient attention mechanisms
    with compression, selection, and sliding window strategies.

    Args:
        args: Training arguments containing model hyperparameters

    Returns:
        NSAModelConfig: Configuration object for NSA model
    """
    # Define key parameters based on size
    if args.size == 'small':
        n_layer = 12
        n_embd = 768
        n_head = 12
        compress_block_size = 16
        compress_stride = 8
        selection_block_size = 32
        num_selected_blocks = 8
        sliding_window_size = 256
    elif args.size == 'medium':
        n_layer = 24
        n_embd = 1024
        n_head = 16
        compress_block_size = 32
        compress_stride = 16
        selection_block_size = 64
        num_selected_blocks = 12
        sliding_window_size = 384
    elif args.size == 'large':
        n_layer = 32
        n_embd = 2048
        n_head = 16
        compress_block_size = 32
        compress_stride = 16
        selection_block_size = 64
        num_selected_blocks = 16
        sliding_window_size = 512
    else:  # xl
        n_layer = 40
        n_embd = 2560
        n_head = 20
        compress_block_size = 64
        compress_stride = 32
        selection_block_size = 128
        num_selected_blocks = 20
        sliding_window_size = 640

    # Create config object
    config = NSAModelConfig(
        # Architecture
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        vocab_size=args.vocab_size,
        block_size=args.block_size,

        # NSA specific parameters
        compress_block_size=compress_block_size,
        compress_stride=compress_stride,
        selection_block_size=selection_block_size,
        num_selected_blocks=num_selected_blocks,
        sliding_window_size=sliding_window_size,

        # Common training parameters
        dropout=args.dropout,
        bias=args.bias,
        use_gradient_checkpointing=getattr(args, 'gradient_checkpointing', True),
        use_fp8=args.use_fp8,
        fp8_tile_size=getattr(args, 'fp8_tile_size', 128),

        # DyT options
        use_dyt=getattr(args, 'use_dyt', False),
        dyt_alpha_init=getattr(args, 'dyt_alpha_init', 0.5),

        # Label smoothing
        label_smoothing=getattr(args, 'label_smoothing', 0.0),
    )

    return config


def create_nsa_model(config):
    """
    Create NSA model instance.

    Args:
        config: NSAModelConfig object with model configuration

    Returns:
        NSAModel: Instantiated NSA model
    """
    return NSAModel(config)


def create_hse_config(args):
    """
    Create configuration for HSE Model (NSA orchestrator + MoE experts).

    HSE (Hierarchical Sparse Experts) combines NSA's selective attention
    with MoE experts and memory management through Scribes and QAP (Query Access Protocol).

    Args:
        args: Training arguments containing model hyperparameters

    Returns:
        HSEConfig: Configuration object for HSE model
    """
    # Size presets similar to NSA
    if args.size == 'small':
        n_layer, n_embd, n_head = 12, 768, 12
        compress_block_size, compress_stride = 16, 8
        selection_block_size, num_selected_blocks = 32, 8
        sliding_window_size = 256
    elif args.size == 'medium':
        n_layer, n_embd, n_head = 24, 1024, 16
        compress_block_size, compress_stride = 32, 16
        selection_block_size, num_selected_blocks = 64, 12
        sliding_window_size = 384
    elif args.size == 'large':
        n_layer, n_embd, n_head = 32, 2048, 16
        compress_block_size, compress_stride = 32, 16
        selection_block_size, num_selected_blocks = 64, 16
        sliding_window_size = 512
    else:  # xl
        n_layer, n_embd, n_head = 40, 2560, 20
        compress_block_size, compress_stride = 64, 32
        selection_block_size, num_selected_blocks = 128, 20
        sliding_window_size = 640

    config = HSEConfig(
        # Architecture
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        vocab_size=args.vocab_size,
        block_size=args.block_size,

        # Orchestrator (NSA)
        compress_block_size=compress_block_size,
        compress_stride=compress_stride,
        selection_block_size=selection_block_size,
        num_selected_blocks=num_selected_blocks,
        sliding_window_size=sliding_window_size,

        # Experts
        num_experts=getattr(args, 'num_experts', 8),
        experts_per_token=getattr(args, 'experts_per_token', 2),

        # Scribes
        scribe_chunk_size=getattr(args, 'scribe_chunk_size', 2048),
        scribe_summary_len=getattr(args, 'scribe_summary_len', 128),

        # QAP budgets
        qap_per_step=getattr(args, 'qap_per_step', 12),
        qap_per_expert=getattr(args, 'qap_per_expert', 6),
        qap_max_queries=getattr(args, 'qap_max_queries', 20),

        # Training
        dropout=args.dropout,
        bias=args.bias,
        use_gradient_checkpointing=getattr(args, 'gradient_checkpointing', True),
        use_fp8=getattr(args, 'use_fp8', False),
        fp8_tile_size=getattr(args, 'fp8_tile_size', 128),
        use_dyt=getattr(args, 'use_dyt', False),
        dyt_alpha_init=getattr(args, 'dyt_alpha_init', 0.5),
        label_smoothing=getattr(args, 'label_smoothing', 0.0),

        # Standard attention extras
        ratio_kv=getattr(args, 'ratio_kv', 8),
        attention_backend=getattr(args, 'attention_backend', None),
    )
    return config


def create_hse_model(config):
    """
    Create HSE model instance.

    Args:
        config: HSEConfig object with model configuration

    Returns:
        HSEModel: Instantiated HSE model
    """
    return HSEModel(config)


def _parse_bool_arg(args, arg_name, default=False):
    """
    Parse boolean argument that might be passed as string.

    Helper function to handle boolean arguments that may come as strings
    from command-line arguments.

    Args:
        args: Arguments object
        arg_name: Name of the argument to parse
        default: Default value if argument not found

    Returns:
        bool: Parsed boolean value
    """
    if not hasattr(args, arg_name):
        return default
    value = getattr(args, arg_name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    return bool(value)


def create_hrm_config(args):
    """
    Create configuration for HRM (Hierarchical Reasoning Model).

    HRM implements a recurrent architecture with hierarchical reasoning
    through segments and cycles, with Adaptive Computation Time (ACT)
    for dynamic computation depth.

    Args:
        args: Training arguments containing model hyperparameters

    Returns:
        HRMConfig: Configuration object for HRM model
    """
    # Define sizes matching the paper's ~27M params for small
    if args.size == 'small':
        n_embd = 512
        n_head = 8
        n_inner = 2048
        max_segments = 8
        cycles_per_segment = 2
        steps_per_cycle = 3
    elif args.size == 'medium':
        n_embd = 768
        n_head = 12
        n_inner = 3072
        max_segments = 8
        cycles_per_segment = 3
        steps_per_cycle = 4
    elif args.size == 'large':
        n_embd = 1024
        n_head = 16
        n_inner = 4096
        max_segments = 10
        cycles_per_segment = 4
        steps_per_cycle = 4
    else:  # xl
        n_embd = 1536
        n_head = 24
        n_inner = 6144
        max_segments = 12
        cycles_per_segment = 4
        steps_per_cycle = 5

    config = HRMConfig(
        # Architecture
        n_layer=1,  # HRM uses recurrence, not stacked layers
        n_embd=n_embd,
        n_head=n_head,
        n_inner=n_inner,
        vocab_size=args.vocab_size,
        block_size=args.block_size,

        # HRM specific parameters
        cycles_per_segment=getattr(args, 'hrm_cycles_per_segment', cycles_per_segment) if hasattr(args, 'hrm_cycles_per_segment') and args.hrm_cycles_per_segment is not None else cycles_per_segment,
        steps_per_cycle=getattr(args, 'hrm_steps_per_cycle', steps_per_cycle) if hasattr(args, 'hrm_steps_per_cycle') and args.hrm_steps_per_cycle is not None else steps_per_cycle,
        max_segments=getattr(args, 'hrm_max_segments', max_segments) if hasattr(args, 'hrm_max_segments') and args.hrm_max_segments is not None else max_segments,
        min_segments=getattr(args, 'min_segments', 1),
        gradient_steps=getattr(args, 'hrm_gradient_steps', -1) if hasattr(args, 'hrm_gradient_steps') and args.hrm_gradient_steps is not None else -1,

        # ACT parameters
        use_act=_parse_bool_arg(args, 'hrm_use_act', True),
        act_epsilon=getattr(args, 'act_epsilon', 0.1),
        ponder_loss_weight=getattr(args, 'ponder_loss_weight', 0.01),
        halt_bias_init=getattr(args, 'halt_bias_init', -2.0),

        # Training parameters
        dropout=args.dropout,
        bias=args.bias,
        use_gradient_checkpointing=False,  # HRM uses 1-step gradient instead
        deq_one_step=getattr(args, 'hrm_deq_one_step', False),
        use_deep_supervision=getattr(args, 'hrm_use_deep_supervision', False),
        n_supervision_segments=getattr(args, 'hrm_n_supervision_segments', 4),
        label_smoothing=getattr(args, 'label_smoothing', 0.0),

        # Learning rate and optimization
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_iters,
        grad_clip=getattr(args, 'grad_clip', 1.0),
    )
    return config


def create_hrm_model(config):
    """
    Create HRM model instance.

    Args:
        config: HRMConfig object with model configuration

    Returns:
        HRM: Instantiated HRM model
    """
    return HRM(config)