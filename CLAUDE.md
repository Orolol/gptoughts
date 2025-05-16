# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

GPToughts is a framework for training and fine-tuning Large Language Models (LLMs), with a focus on GPU optimizations for hobby-scale training. The repository supports multiple model architectures including GPT-style autoregressive models, DeepSeek models, and LLaDA (Large Language Diffusion with mAsking) models which use a diffusion-based approach.

## Key Commands

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up Hugging Face token (needed for accessing some models)
export HF_TOKEN=your_huggingface_token
```

### Training

Basic training:
```bash
python run_train.py --model_type [llada|deepseek|gpt] --size [small|medium|large] --batch_size NUM --block_size NUM
```

With FP8 precision (requires H100/H200 GPU):
```bash
python run_train.py --model_type llada --size medium --use_fp8 --batch_size 32 --block_size 4096
```

Using the optimization script (recommended):
```bash
# Apply all optimizations and train
./optimize.sh --all --train --model llada --size medium

# Apply specific optimizations
./optimize.sh --memory --cuda --auto-batch --train --model llada --size small
```

### Optimization Options

View available optimization options:
```bash
./optimize.sh --help
```

Common options:
- `--fp8`: Enable FP8 precision (H100/H200/Ada Lovelace GPUs)
- `--memory`: Memory optimization
- `--cuda`: CUDA optimizations
- `--attention`: Enable optimized attention operations
- `--auto-batch`: Automatically determine optimal batch size
- `--compile`: Use torch.compile for speed
- `--recover-nan`: Add stabilization for NaN issues
- `--all`: Enable all optimizations
- `--train`: Start training after applying optimizations

### GPU Statistics

```bash
# Show GPU statistics
./optimize.sh --stats
```

## Project Architecture

### Core Components

1. **Models**:
   - `models/models/`: Core model implementations (GPT-style)
   - `models/deepseek/`: DeepSeek model implementations (with MLA and MTP)
   - `models/llada/`: LLaDA (diffusion-based) model implementations
   - `models/blocks/`: Shared building blocks (attention, MLP, etc.)

2. **Training**:
   - `train/train.py`: Main training loop with GPU optimizations
   - `train/train_utils.py`: Utilities for training (metrics, checkpoints)
   - `train/train_moe.py`: Training for Mixture of Experts models
   - `train/memory_utils.py`: Memory management utilities

3. **Data Loading**:
   - `data/data_loader.py`: Current data loader implementation
   - `data/data_loader_legacy.py`: Legacy data loader
   - `data/data_loader_llada.py`: Data loader for LLaDA models
   - `data/data_loader_original.py`: Original data loader implementation

4. **Optimization**:
   - `optimization/fp8_optim.py`: FP8 precision optimizations
   - `optimization/memory_optim.py`: Memory optimizations
   - `optimization/cuda_optim.py`: CUDA optimizations
   - `optimization/training_optim.py`: Training-specific optimizations

5. **Entry Points**:
   - `run_train.py`: Main entry point for training
   - `optimize.sh`: Script for applying optimizations

### Model Architectures

#### DeepSeek Model
DeepSeek uses two innovative techniques:
1. **Multi-head Latent Attention (MLA)**: Compresses key-value embeddings for efficient memory usage
2. **Multi-Token Prediction (MTP)**: Predicts multiple future tokens for enhanced training and speculative decoding

#### LLaDA Model
The LLaDA model (Large Language Diffusion with mAsking) uses a diffusion-based approach:
1. Transformer encoder with bidirectional attention (no causal masking)
2. During training, applies random masking and predicts original tokens
3. For inference, uses an iterative demasking process

## Development Workflow

1. **Training a Model**:
   ```bash
   ./optimize.sh --all --train --model llada --size small
   ```

2. **Resuming Training**:
   ```bash
   python run_train.py --model_type llada --size small --init_from resume --output_dir out
   ```

3. **Text Generation**:
   - Models automatically generate sample outputs every 500 iterations during training
   - For LLaDA models, generation uses an iterative demasking process
   - For DeepSeek models with MTP, speculative decoding is available for faster generation

## Important Implementation Details

1. **GPU Optimization Layers**:
   - The code implements multiple layers of GPU optimization
   - These can be activated independently or together via `optimize.sh`
   - The codebase auto-detects GPU capabilities and enables compatible optimizations

2. **Model Sizes**:
   - `small`: For experimentation (fits on consumer GPUs)
   - `medium`: Mid-sized models (requires high-end consumer or workstation GPUs)
   - `large`: Full-sized models (requires enterprise/datacenter GPUs)

3. **Precision Options**:
   - FP32: Standard precision (slow but stable)
   - FP16: Mixed precision (faster, may be unstable)
   - BF16: Preferred for Ampere+ GPUs (better numerical stability)
   - FP8: Fastest but requires H100/H200/Ada Lovelace GPUs

4. **Handling NaN Issues**:
   - For training stability issues, use `--recover-nan` option
   - The code includes built-in NaN detection and recovery mechanisms

## Tips and Best Practices

1. **Memory Management**:
   - Use `--auto-batch` to determine optimal batch size for your GPU
   - For large models, increase gradient accumulation steps instead of batch size
   - Use `--memory` optimization for better memory management

2. **Performance Optimization**:
   - Enable `--compile` for PyTorch 2.0+ speed improvements
   - Use `--attention` for Flash Attention and other attention optimizations
   - If your GPU supports it, enable `--fp8` for maximum performance

3. **Stability**:
   - If training becomes unstable, try reducing learning rate or enabling `--recover-nan`
   - For LLaDA models, adjust `router_z_loss_coef` to a smaller value (e.g., 0.0001)
   - Use gradient clipping with a conservative value (e.g., 1.0)

## Code Quality Principles

1. **DRY (Don't Repeat Yourself)**:
   - Avoid duplicating code across different files or modules
   - Extract common functionality into shared utility functions
   - Reuse existing components rather than creating new ones with similar functionality
   - When extending functionality, inherit from existing classes rather than copying code

2. **Code Organization**:
   - Keep the codebase tidy with a clear, logical file structure
   - Minimize the number of files; don't create separate files for very small components
   - Group related functionality in the same file or module
   - Follow the existing modular architecture pattern

3. **Simplicity and Readability**:
   - Keep code simple and straightforward
   - Prioritize readability and maintainability over cleverness
   - Use descriptive variable and function names
   - Add clear docstrings that explain purpose and usage

4. **Factorization**:
   - Factor out common patterns and functionality
   - Use composition and inheritance effectively to avoid redundancy
   - Create abstractions only when they simplify the codebase, not for their own sake
   - Balance abstraction with concrete implementations for clarity

5. **Changes and Extensions**:
   - When adding new features, extend existing structures when possible
   - Maintain backward compatibility with existing interfaces
   - Use adapter patterns to integrate new functionality with minimal changes
   - Ensure new components follow the same design patterns as existing ones