# MLA with Selective Attention

This document describes the MLA-Selective model, which combines Multi-head Latent Attention (MLA) with a selective attention mechanism that dynamically selects which tokens to attend to based on importance scores.

## Overview

The MLA-Selective model extends the standard MLA architecture by adding a token selection mechanism that:
- Computes importance scores for each token using a learned scoring network
- Selects a subset of tokens based on these scores
- Applies attention only to the selected tokens

This approach can lead to:
- **Improved efficiency**: By attending to fewer tokens, computational cost is reduced
- **Better focus**: The model can learn to focus on the most relevant tokens
- **Adaptability**: Different selection ratios can be used for different tasks

## Architecture

### Components

1. **Importance Scoring Network**: A small MLP that computes importance scores for each token
   ```python
   self.importance_score = nn.Sequential(
       nn.Linear(dim, dim // 4),
       nn.ReLU(),
       nn.Linear(dim // 4, 1)
   )
   ```

2. **Token Selection**: Three methods are supported:
   - **top_k**: Select the top-k most important tokens
   - **threshold**: Select tokens above a learned threshold
   - **gumbel**: Use Gumbel-Softmax for differentiable selection

3. **Selective MLA**: Apply standard MLA attention only to selected tokens

### Configuration Parameters

- `selection_ratio`: Fraction of tokens to select (0.0 to 1.0, default: 0.5)
- `selection_method`: Method for token selection ('top_k', 'threshold', 'gumbel')
- `selection_temperature`: Temperature for Gumbel-Softmax selection (default: 1.0)

## Usage

### Training

To train an MLA-Selective model:

```bash
./train_mla_selective.sh small 8 2048 out_mla_selective 0 0.5 top_k
```

Parameters:
1. Model size (small/medium/large/xl)
2. Batch size
3. Block size (sequence length)
4. Output directory
5. Use FP8 (0/1)
6. Selection ratio (0.0-1.0)
7. Selection method (top_k/threshold/gumbel)

### Python API

```python
from models.models import create_mla_selective_model

# Create model
model = create_mla_selective_model(
    size='small',
    vocab_size=50304,
    block_size=2048,
    selection_ratio=0.5,
    selection_method='top_k'
)

# Use like any other model
logits, loss = model(input_ids, targets)
```

## Implementation Details

### Token Selection Process

1. **Compute Importance**: For each token in the sequence, compute an importance score
2. **Select Tokens**: Based on the selection method, choose which tokens to attend to
3. **Apply Mask**: Mask out non-selected tokens in the attention computation

### Selection Methods

#### Top-K Selection
- Selects the k tokens with highest importance scores
- k = max(1, int(seq_len * selection_ratio))
- Deterministic and differentiable through scores

#### Threshold Selection
- Selects tokens with importance above a dynamic threshold
- Threshold is computed as the (1-selection_ratio) quantile of scores
- Can result in variable number of selected tokens

#### Gumbel Selection
- Uses Gumbel-Softmax for differentiable discrete selection
- Allows gradient flow through the selection process
- Temperature controls the sharpness of selection

### Integration with MLA

The selective attention mechanism is integrated into the MLA forward pass:

1. Compute Q, K, V projections as normal
2. Compute importance scores and select tokens
3. Apply selection mask to attention scores
4. Continue with standard MLA computation

## Performance Considerations

- **Memory**: Selective attention reduces memory usage in attention computation
- **Speed**: Fewer tokens to attend to means faster attention computation
- **Quality**: Proper tuning of selection_ratio is important for maintaining model quality

## Experimental Results

(To be added based on experimental results)

## Future Work

- Dynamic selection ratio based on sequence complexity
- Hierarchical selection (different ratios for different layers)
- Learned selection policies
- Integration with other efficiency techniques