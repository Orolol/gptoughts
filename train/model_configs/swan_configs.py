"""SWAN and SWA_MLA model configuration builders."""

from models.models.swan_model import SWANConfig
from models.models.swa_mla_model import SWAMLAConfig


def create_swan_config(args):
    """Create SWAN (Sliding Window Attention Network) configuration based on model size.

    SWAN uses a hybrid attention mechanism combining global and local (sliding window)
    attention layers to balance long-range dependencies with computational efficiency.

    Args:
        args: Training arguments containing:
            - size: Model size ('small', 'medium', 'large', 'xl')
            - vocab_size: Vocabulary size
            - block_size: Maximum sequence length
            - dropout: Dropout probability
            - bias: Whether to use bias in linear layers
            - n_layer: Optional override for number of layers
            - n_head: Optional override for number of attention heads
            - n_embd: Optional override for embedding dimension
            - global_layers_per_cycle: Number of global attention layers per cycle (default: 1)
            - local_layers_per_cycle: Number of local (SWA) layers per cycle (default: 3)
            - swa_window: Sliding window size for local attention (default: 512)
            - ratio_kv: KV cache ratio (default: 1)
            - attention_backend: Attention implementation backend
            - use_gradient_checkpointing: Enable gradient checkpointing (default: True)
            - rope_theta: RoPE theta parameter (default: 10000.0)
            - logit_scale_base: Base scale for logit scaling (default: 128.0)
            - logit_scale_window: Window size for logit scaling (default: 128)
            - logit_scale_offset: Offset for logit scaling (default: 0)
            - logit_scale_min: Minimum logit scale (default: 1.0)
            - logit_scale_max: Maximum logit scale (default: None)
            - apply_logit_scale_during_training: Apply scaling during training (default: False)
            - label_smoothing: Label smoothing factor (default: 0.0)

    Returns:
        SWANConfig: Configuration for SWAN model
    """
    size_defaults = {
        'small': dict(n_layer=16, n_embd=1024, n_head=16),
        'medium': dict(n_layer=24, n_embd=1536, n_head=16),
        'large': dict(n_layer=28, n_embd=2048, n_head=24),
        'xl': dict(n_layer=32, n_embd=4096, n_head=32),
    }
    size_key = getattr(args, 'size', 'medium')
    size_config = size_defaults.get(size_key, size_defaults['medium'])

    n_layer = getattr(args, 'n_layer', None)
    if n_layer is None:
        n_layer = size_config['n_layer']
    n_head = getattr(args, 'n_head', None)
    if n_head is None:
        n_head = size_config['n_head']
    n_embd = getattr(args, 'n_embd', None)
    if n_embd is None:
        n_embd = size_config['n_embd']

    global_layers = getattr(args, 'global_layers_per_cycle', None)
    if global_layers is None:
        global_layers = 1
    local_layers = getattr(args, 'local_layers_per_cycle', None)
    if local_layers is None:
        local_layers = 3
    swa_window = getattr(args, 'swa_window', None)
    if swa_window is None:
        swa_window = 512

    logit_scale_base = getattr(args, 'logit_scale_base', None)
    if logit_scale_base is None:
        logit_scale_base = 128.0
    logit_scale_window = getattr(args, 'logit_scale_window', None)
    if logit_scale_window is None:
        logit_scale_window = 128
    logit_scale_offset = getattr(args, 'logit_scale_offset', None)
    if logit_scale_offset is None:
        logit_scale_offset = 0
    logit_scale_min = getattr(args, 'logit_scale_min', None)
    if logit_scale_min is None:
        logit_scale_min = 1.0
    logit_scale_max = getattr(args, 'logit_scale_max', None)

    config = SWANConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=args.dropout,
        bias=args.bias,
        ratio_kv=getattr(args, 'ratio_kv', 1),
        attention_backend=getattr(args, 'attention_backend', None),
        use_gradient_checkpointing=getattr(args, 'use_gradient_checkpointing', getattr(args, 'gradient_checkpointing', True)),
        global_layers_per_cycle=global_layers,
        local_layers_per_cycle=local_layers,
        swa_window=swa_window,
        rope_theta=getattr(args, 'rope_theta', 10000.0),
        logit_scale_base=logit_scale_base,
        logit_scale_window=logit_scale_window,
        logit_scale_offset=logit_scale_offset,
        logit_scale_min=logit_scale_min,
        logit_scale_max=logit_scale_max,
        apply_logit_scale_during_training=getattr(args, 'apply_logit_scale_during_training', False),
        label_smoothing=getattr(args, 'label_smoothing', 0.0),
    )

    return config


def create_swa_mla_config(args):
    """Create SWA_MLA (Sliding Window Attention + Multi-head Latent Attention) configuration.

    SWA_MLA combines efficient sliding window attention with MLA's latent compression
    to achieve both computational efficiency and strong modeling capacity.

    Args:
        args: Training arguments containing:
            - size: Model size ('small', 'medium', 'large', 'xl')
            - vocab_size: Vocabulary size
            - block_size: Maximum sequence length
            - dropout: Dropout probability
            - bias: Whether to use bias in linear layers
            - n_layer: Optional override for number of layers
            - n_head: Optional override for number of attention heads
            - n_embd: Optional override for embedding dimension
            - swa_layers_per_cycle: Number of SWA layers per cycle (default: local_layers_per_cycle or 2)
            - mla_layers_per_cycle: Number of MLA layers per cycle (default: global_layers_per_cycle or 1)
            - swa_window: Sliding window size (default: 256)
            - ratio_kv: KV cache ratio (default: 1)
            - attention_backend: Attention implementation backend
            - use_gradient_checkpointing: Enable gradient checkpointing (default: True)
            - use_dyt: Enable DYT (Dynamic Token) optimization (default: False)
            - dyt_alpha_init: Initial DYT alpha value (default: 0.5)
            - rope_theta: RoPE theta parameter (default: 10000.0)
            - logit_scale_base: Base scale for logit scaling (default: None)
            - logit_scale_window: Window size for logit scaling (default: 128)
            - logit_scale_offset: Offset for logit scaling (default: 0)
            - logit_scale_min: Minimum logit scale (default: 1.0)
            - logit_scale_max: Maximum logit scale (default: None)
            - apply_logit_scale_during_training: Apply scaling during training (default: False)
            - mla_q_lora_rank: Query LoRA rank for MLA (default: 0)
            - mla_kv_lora_rank: Key-Value LoRA rank for MLA (default: 512)
            - mla_qk_nope_head_dim: Query-Key non-positional head dimension (default: 128)
            - mla_qk_rope_head_dim: Query-Key RoPE head dimension (default: 64)
            - mla_v_head_dim: Value head dimension (default: 128)
            - mla_attn_impl: MLA attention implementation ('absorb', etc.) (default: 'absorb')
            - mla_rope_scaling: RoPE scaling configuration (default: None)
            - mla_rope_factor: RoPE scaling factor (default: 1.0)
            - mla_mscale: MLA attention scaling factor (default: 1.0)
            - use_fp8: Enable FP8 precision (default: False)
            - fp8_mla_params: Use FP8 for MLA parameters (default: False)
            - fp8_tile_size: Tile size for FP8 operations (default: 128)
            - label_smoothing: Label smoothing factor (default: 0.0)
            - devices: Number of devices for distributed training (default: 1)

    Returns:
        SWAMLAConfig: Configuration for SWA_MLA model
    """
    size_defaults = {
        'small': dict(n_layer=12, n_embd=1024, n_head=16),
        'medium': dict(n_layer=24, n_embd=1536, n_head=16),
        'large': dict(n_layer=28, n_embd=2048, n_head=24),
        'xl': dict(n_layer=32, n_embd=4096, n_head=32),
    }
    size_key = getattr(args, 'size', 'medium')
    size_config = size_defaults.get(size_key, size_defaults['medium'])

    n_layer = getattr(args, 'n_layer', None) or size_config['n_layer']
    n_head = getattr(args, 'n_head', None) or size_config['n_head']
    n_embd = getattr(args, 'n_embd', None) or size_config['n_embd']

    swa_layers = getattr(args, 'swa_layers_per_cycle', None)
    if swa_layers is None:
        swa_layers = getattr(args, 'local_layers_per_cycle', 2)
    mla_layers = getattr(args, 'mla_layers_per_cycle', None)
    if mla_layers is None:
        mla_layers = getattr(args, 'global_layers_per_cycle', 1)

    logit_scale_base = getattr(args, 'logit_scale_base', None)
    logit_scale_window = getattr(args, 'logit_scale_window', None)
    if logit_scale_window is None:
        logit_scale_window = 128
    logit_scale_offset = getattr(args, 'logit_scale_offset', None)
    if logit_scale_offset is None:
        logit_scale_offset = 0
    logit_scale_min = getattr(args, 'logit_scale_min', None)
    if logit_scale_min is None:
        logit_scale_min = 1.0
    logit_scale_max = getattr(args, 'logit_scale_max', None)

    config = SWAMLAConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=args.dropout,
        bias=args.bias,
        ratio_kv=getattr(args, 'ratio_kv', 1),
        attention_backend=getattr(args, 'attention_backend', None),
        use_gradient_checkpointing=getattr(args, 'use_gradient_checkpointing', getattr(args, 'gradient_checkpointing', True)),
        use_dyt=getattr(args, 'use_dyt', False),
        dyt_alpha_init=getattr(args, 'dyt_alpha_init', 0.5),
        swa_layers_per_cycle=swa_layers,
        mla_layers_per_cycle=mla_layers,
        swa_window=getattr(args, 'swa_window', 256),
        rope_theta=getattr(args, 'rope_theta', 10000.0),
        logit_scale_base=logit_scale_base,
        logit_scale_window=logit_scale_window,
        logit_scale_offset=logit_scale_offset,
        logit_scale_min=logit_scale_min,
        logit_scale_max=logit_scale_max,
        apply_logit_scale_during_training=getattr(args, 'apply_logit_scale_during_training', False),
        q_lora_rank=getattr(args, 'mla_q_lora_rank', 0),
        kv_lora_rank=getattr(args, 'mla_kv_lora_rank', 512),
        qk_nope_head_dim=getattr(args, 'mla_qk_nope_head_dim', 128),
        qk_rope_head_dim=getattr(args, 'mla_qk_rope_head_dim', 64),
        v_head_dim=getattr(args, 'mla_v_head_dim', 128),
        attn_impl=getattr(args, 'mla_attn_impl', 'absorb'),
        world_size=getattr(args, 'devices', 1) if isinstance(getattr(args, 'devices', -1), int) else 1,
        rope_scaling=getattr(args, 'mla_rope_scaling', None),
        rope_factor=getattr(args, 'mla_rope_factor', 1.0),
        mscale=getattr(args, 'mla_mscale', 1.0),
        use_fp8=getattr(args, 'use_fp8', False),
        fp8_mla_params=getattr(args, 'fp8_mla_params', False),
        fp8_tile_size=getattr(args, 'fp8_tile_size', 128),
        label_smoothing=getattr(args, 'label_smoothing', 0.0),
        use_mla_selective=getattr(args, 'use_mla_selective', False),
        selection_head_idx=getattr(args, 'mla_selection_head_idx', 0),
        swa_sink_size=getattr(args, 'swa_sink_size', 4),
    )

    rope_scaling = getattr(args, 'mla_rope_scaling', None)
    if rope_scaling is not None and isinstance(rope_scaling, dict):
        config.rope_scaling = rope_scaling

    return config