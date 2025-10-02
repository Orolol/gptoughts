# HRM (Hierarchical Reasoning Model) Implementation

## Overview

This document describes the implementation of the Hierarchical Reasoning Model (HRM) in the GPToughts framework. HRM is a brain-inspired recurrent architecture designed for complex reasoning tasks, achieving remarkable performance with minimal parameters and training data.

## Key Features

### 1. **Hierarchical Architecture**
- **Two-level processing**: High-level (H) and Low-level (L) modules
- **Temporal separation**: L-module updates every timestep, H-module updates every T steps
- **Hierarchical convergence**: Prevents premature convergence of standard RNNs

### 2. **Efficient Training**
- **1-step gradient approximation**: O(1) memory instead of O(T) for BPTT
- **Deep supervision**: Multiple training segments with detached gradients
- **No pre-training required**: Learns from scratch with ~1000 examples

### 3. **Adaptive Computation**
- **ACT with Q-learning**: Dynamic resource allocation based on task complexity
- **Ponder loss**: Regularizes computation time
- **Inference-time scaling**: Can use more computation at inference for better results

## Model Sizes

| Size   | Parameters | Embedding | Heads | FFN Size | Use Case |
|--------|------------|-----------|-------|----------|----------|
| Small  | ~27M       | 512       | 8     | 2048     | Research, small datasets |
| Medium | ~97M       | 768       | 12    | 3072     | General purpose |
| Large  | ~138M      | 1024      | 16    | 4096     | Complex reasoning |
| XL     | ~250M      | 1536      | 24    | 6144     | Maximum capability |

## Usage

### Training with Script

```bash
# Train small HRM model (default settings)
./train_hrm.sh

# Train medium model with custom parameters
./train_hrm.sh medium 16 1024 out_hrm_medium 0

# Parameters: size batch_size block_size output_dir use_fp8
```

### Python API

```python
from models.models.hrm_model import create_hrm_model, HRMConfig

# Create model with preset size
model = create_hrm_model(
    size='small',
    vocab_size=50304,
    block_size=900,
    dropout=0.1
)

# Custom configuration
config = HRMConfig(
    n_embd=512,
    n_head=8,
    cycles_per_segment=2,
    steps_per_cycle=3,
    max_segments=8,
    use_act=True,
    use_deep_supervision=True
)
model = HRM(config)
```

### Training with Deep Supervision

```python
from models.models.hrm_model import train_with_deep_supervision

# Training loop with deep supervision
train_with_deep_supervision(
    model=model,
    dataloader=train_loader,
    optimizer=optimizer,
    config=config
)
```

## Architecture Details

### Hierarchical Processing

```
Input → L-module (T steps) → H-module (1 update) → ... → Output
         ↑___________________|
         Reset and new context
```

### 1-Step Gradient Flow

```python
# Forward without gradient for N*T-1 steps
with torch.no_grad():
    for i in range(total_steps - 1):
        z_L = L_module(z_L, z_H, x)
        if (i + 1) % T == 0:
            z_H = H_module(z_H, z_L)

# Only compute gradient for final step
z_L = L_module(z_L, z_H, x)  # With gradient
z_H = H_module(z_H, z_L)      # With gradient
```

## Performance Benchmarks

Based on the paper's results with ~1000 training examples:

| Task | HRM (~27M params) | o3-mini-high | Claude 3.7 | 
|------|-------------------|--------------|------------|
| ARC-AGI-1 | **40.3%** | 34.5% | 21.2% |
| Sudoku-Extreme | **55.0%** | 0% | 0% |
| Maze-Hard (30×30) | **74.5%** | 0% | 0% |

## Key Innovations

### 1. Hierarchical Convergence
- L-module converges to local equilibrium within each cycle
- H-module updates perturb this equilibrium, preventing stagnation
- Maintains computational activity over N×T steps

### 2. Biological Plausibility
- Inspired by cortical hierarchies and neural oscillations
- Temporal separation mimics theta-gamma coupling
- Local learning rules instead of global BPTT

### 3. Efficiency
- Constant memory footprint during training
- No need for large-scale pre-training
- Excellent data efficiency (1000 examples)

## Configuration Parameters

### Core Parameters
- `n_embd`: Model dimension (512 for small)
- `n_head`: Number of attention heads
- `n_inner`: FFN inner dimension
- `block_size`: Maximum sequence length

### HRM-Specific Parameters
- `cycles_per_segment`: N - high-level cycles per segment (default: 2)
- `steps_per_cycle`: T - low-level steps per cycle (default: 3)
- `max_segments`: Maximum segments for ACT (default: 8)
- `min_segments`: Minimum segments before halting (default: 1)

### ACT Parameters
- `use_act`: Enable adaptive computation (default: True)
- `act_epsilon`: Exploration rate for Q-learning (default: 0.1)
- `ponder_loss_weight`: Weight for computation regularization (default: 0.01)
- `halt_bias_init`: Initial bias for halt decision (default: -2.0)

### Training Parameters
- `use_deep_supervision`: Enable deep supervision (default: True)
- `n_supervision_segments`: Number of supervision segments (default: 4)
- `label_smoothing`: Label smoothing factor (default: 0.0)

## Implementation Files

- `models/models/hrm_model.py`: Main HRM implementation
- `train_hrm.sh`: Training script with optimal settings
- `train/lightning_module.py`: PyTorch Lightning integration
- `test_hrm.py`: Test suite for verification

## Tips for Training

1. **Start with small models**: HRM is very parameter-efficient
2. **Use deep supervision**: Significantly improves training stability
3. **Monitor ponder cost**: Should adapt based on task complexity
4. **Adjust ACT parameters**: Lower epsilon for more exploitation
5. **Use gradient clipping**: Recommended value is 1.0

## Future Improvements

1. **Multi-modal support**: Extend to vision and audio
2. **Continuous learning**: Adapt to new tasks without forgetting
3. **Interpretability**: Analyze learned reasoning strategies
4. **Hardware optimization**: Custom kernels for hierarchical updates

## References

- Original paper: "Hierarchical Reasoning Model" (arXiv:2506.21734v1)
- Authors: Guan Wang, Jin Li, et al. - Sapient Intelligence, Singapore
- Implementation based on paper specifications and biological principles