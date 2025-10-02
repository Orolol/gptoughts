# FP8 MLA Integration Guide

## Overview

This document describes how the DeepSeek-V3 FP8 training approach has been integrated into the MLA models within the GPToughts framework.

## Implementation Details

### 1. Core Components

#### FP8 Quantization Module (`optimization/fp8_mla.py`)
- **FP8Quantizer**: Implements tile-wise (1×128) and block-wise (128×128) quantization
- **FP8LinearMLA**: Linear layer with FP8 computation and high-precision accumulation
- Mixed precision utilities

#### FP8 MLA Block (`models/blocks/mla_fp8.py`)
- **MLA_FP8**: MLA attention block with integrated FP8 support
- Low-precision caching for inference
- Mixed precision attention computation

#### FP8 Training Utilities (`optimization/fp8_deepseek_trainer.py`)
- **FP8AdamW**: Optimizer with BF16 moment storage
- **FP8MixedPrecisionTrainer**: Training context and utilities
- GPU architecture detection and configuration

### 2. Integration Points

#### Model Creation
In `models/models/mla_model.py`:
- MLAModelBlock checks `config.use_fp8` to select between MLA and MLA_FP8
- Configuration includes FP8-specific parameters

#### Lightning Module
In `train/lightning_module.py`:
- Passes FP8 parameters to model configuration
- Conditionally uses FP8AdamW optimizer when `use_fp8=True`
- Separates parameters into high/low precision groups

#### Command Line Arguments
In `run_train.py`:
- `--use_fp8`: Enable FP8 precision
- `--fp8_mla_params`: Use FP8 for MLA linear layers (default: False for stability)
- `--fp8_tile_size`: Quantization tile size (default: 128)

### 3. Usage

#### Basic Training with FP8
```bash
python run_train.py \
    --model_type mla \
    --size small \
    --use_fp8 \
    --batch_size 8 \
    --block_size 2048
```

#### Using the Training Script
```bash
./train_mla_fp8_deepseek.sh small 8 2048 out_mla_fp8 1
```

#### Python API
```python
from models.models.mla_model import MLAModel, MLAModelConfig

# Create FP8-enabled MLA model
config = MLAModelConfig(
    n_layer=12,
    n_embd=768,
    n_head=12,
    use_fp8=True,           # Enable FP8 blocks
    fp8_params=True,        # Use FP8 computation
    fp8_mla_params=False,   # Keep MLA params in FP16
    fp8_tile_size=128       # Tile size for quantization
)

model = MLAModel(config)
```

### 4. Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_fp8` | False | Master switch to use FP8-optimized MLA blocks |
| `fp8_params` | False | Enable FP8 computation (vs storage) |
| `fp8_mla_params` | False | Use FP8 for MLA linear layers (keep False for stability) |
| `fp8_tile_size` | 128 | Tile size for fine-grained quantization |

### 5. Mixed Precision Strategy

Following DeepSeek's approach, these components remain in high precision:
- Embeddings
- Normalization layers (RMSNorm, LayerNorm)
- Positional encoding (RoPE)
- Attention computation
- Master weights and gradients

FP8 is used for:
- Linear layer computations
- Activation caching (during inference)
- Optimizer moments (BF16 instead of FP32)

### 6. Performance Characteristics

Expected benefits:
- **Memory**: ~50% reduction in activation memory
- **Compute**: Up to 2x theoretical speedup on H100/H200
- **Optimizer**: ~50% reduction in moment storage

Trade-offs:
- Slightly reduced precision (typically <0.25% loss difference)
- Requires compatible hardware (H100, H200, or Ada GPUs)
- Additional quantization overhead (~5%)

### 7. Current Status

✅ **Implemented**:
- Fine-grained quantization (tile/block-wise)
- FP8-optimized MLA blocks
- Mixed precision framework
- Low-precision optimizer moments
- Integration with existing training pipeline

⚠️ **Known Issues**:
- Scale factor broadcasting in FP8LinearMLA needs fixing
- Custom CUDA kernels not yet implemented (using PyTorch ops)
- Limited to E4M3 format (E5M2 support pending)

🚧 **Future Work**:
- Custom CUDA kernels for true high-precision accumulation
- Dynamic loss scaling for FP8 gradients
- Microscaling format support for next-gen GPUs
- Full transformer_engine integration

### 8. Testing

Run the integration test:
```bash
python test_fp8_integration.py
```

This verifies:
- Correct block instantiation based on configuration
- Forward/backward pass functionality
- Parameter count consistency
- Gradient flow

### 9. Debugging

Enable debug mode:
```bash
export CUDA_LAUNCH_BLOCKING=1
export NVTE_DEBUG=1
export TORCH_SHOW_CPP_STACKTRACES=1
```

Common issues:
- **Import errors**: Ensure all dependencies are installed
- **GPU compatibility**: Check with `nvidia-smi`
- **Memory errors**: Reduce batch size or enable gradient checkpointing
- **Convergence**: Start with BF16 to validate, then enable FP8