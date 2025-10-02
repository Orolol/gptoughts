# MLA with Selective Attention

This document describes the MLA-Selective model, which combines Multi-head Latent Attention (MLA) with the selective attention mechanism from "Selective Attention Improves Transformer" (arXiv:2410.02703).

## Overview

The MLA-Selective model extends the standard MLA architecture by adding a parameter-free token selection mechanism that:
- Reuses one attention head's output as a selection function
- Applies masking to reduce attention to unneeded tokens
- Provides memory savings without adding parameters

This approach leads to:
- **Memory efficiency**: 16-47x less memory for attention (for context sizes 512-2048)
- **Equivalent performance**: Similar to models with 2x more attention heads/parameters
- **No additional parameters**: Reuses existing attention weights
- **Trade-off**: Disables Flash Attention/SDPA optimizations when enabled

## Architecture

### Components

1. **Selection Head**: Uses one existing attention head's output for masking
   - No additional parameters
   - Configurable via `selection_head_idx` (default: 0)

2. **Masking Process**:
   - Apply ReLU to attention weights (only reduce, never boost attention)
   - Zero out first column (preserve BOS token)
   - Zero out diagonal (prevent self-masking)
   - Accumulate masks causally across sequence

3. **Integration**: Selective masking is applied to attention logits before softmax

### Configuration Parameters

- `selection_head_idx`: Which attention head to use for selection (default: 0)

## Usage

### Training

To train an MLA-Selective model:

```bash
./train_mla_selective.sh small 8 2048 out_mla_selective 0
```

Parameters:
1. Model size (small/medium/large/xl)
2. Batch size
3. Block size (sequence length)
4. Output directory
5. Use FP8 (0/1)

### Python API

```python
from models.models import create_mla_selective_model

# Create model with selective attention (always enabled in MLASelective)
model = create_mla_selective_model(
    size='small',
    vocab_size=50304,
    block_size=2048,
    selection_head_idx=0  # Which head to use for selection
)

# Use like any other model
logits, loss = model(input_ids, targets)
```

## Implementation Details

### Selective Attention Algorithm

Following the paper "Selective Attention Improves Transformer":

1. **Compute standard attention**: Calculate attention weights normally
2. **Extract selection weights**: Use attention weights from one head as selection function
3. **Apply constraints**:
   - ReLU: Only allow positive masking values (reduce attention, not boost)
   - Preserve BOS: Zero out first column to maintain start token
   - No self-masking: Zero out diagonal
4. **Accumulate mask**: Sum masks from previous tokens causally
5. **Apply to attention**: Subtract accumulated mask from attention logits

### Formula

```
SelectiveAttention(Q, K, V) = softmax((QK^T / √d_k) - F) V
```

Where F is the accumulated masking matrix computed by summing previous tokens' selection values.

### Performance Considerations

MLASelective always uses selective attention:
- Memory usage: 16-47x reduction in attention memory
- Speed: Optimal performance with PyTorch FlexAttention (when available)
- Quality: Equivalent to models with 2x parameters
- **FlexAttention**: Custom attention patterns compiled to efficient kernels
- **Fallback**: Uses SDPA without selective masking when FlexAttention unavailable

For standard MLA without selective attention, use the regular MLA model.

### Implementation Details

The implementation uses PyTorch's FlexAttention API which:
- Compiles custom attention patterns to efficient CUDA kernels
- Provides performance comparable to handwritten kernels
- Supports block-sparse attention patterns
- Integrates seamlessly with PyTorch's autograd

## Experimental Results

Based on the original paper:
- Memory reduction: 16x (512 tokens), 25x (1024 tokens), 47x (2048 tokens)
- Performance: Equivalent to transformers with 2x more attention heads/parameters
- No additional parameters required

## Future Work

- Custom CUDA/Triton kernels to enable selective attention with Flash Attention
- FlashAttention 3 integration when custom masking is supported
- Hybrid approach: compute only selection head manually, use SDPA for others
- Per-layer selection head configuration