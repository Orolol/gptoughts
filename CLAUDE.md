# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

GPToughts is a framework for training and fine-tuning Large Language Models (LLMs), with a focus on GPU optimizations for hobby-scale training. The repository supports multiple model architectures including GPT-style autoregressive models, DeepSeek models, and LLaDA (Large Language Diffusion with mAsking) models which use a diffusion-based approach.

## Project Architecture

The codebase is organized into several key components:

1. **Model Architectures**:
   - Standard autoregressive models (GPT-style)
   - DeepSeek models with various adapters
   - LLaDA models (diffusion-based approach)
   - MLA (Multi-head Latent Attention) models

2. **Training Infrastructure**:
   - PyTorch Lightning modules (`train/lightning_module.py`)
   - Optimized data loaders with different implementations
   - Memory optimization utilities

3. **GPU Optimizations**:
   - FP8 precision support for compatible hardware
   - Memory usage optimizations
   - CUDA-specific performance enhancements
   - Training workflow optimizations

## Current Focus and Development

- We are currently working on MLA model architecture and related blocks
- LLaDA implementation (diffusion-based approach to language modeling)
- Numerical stability improvements for training with FP8 precision

## Development Environment

```bash
# Use the recommended Python environment
pyenv activate 5090
```

## Key Commands

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up Hugging Face token (needed for accessing some models)
export HF_TOKEN=your_huggingface_token
```

### Training Models

```bash
# Basic training command
python run_train.py --model_type [model_type] --size [size] --batch_size [batch_size] --block_size [block_size]

# Available model types: gpt, deepseek, llada, mla
# Available sizes: small, medium, large, xl

# Using simplified training script (for MLA models with stability optimizations)
./train_simplified.sh [size] [batch_size] [block_size] [output_dir] [use_fp8]

# Example: Train small MLA model with batch size 8 and sequence length 2048
./train_simplified.sh small 8 2048 out_mla 0
```

### GPU Optimization

```bash
# View all optimization options
./optimize.sh --help

# Enable all optimizations
./optimize.sh --all

# Specific optimizations
./optimize.sh --memory --cuda

# Recovery from numerical instability
./emergency_training.sh [checkpoint_dir]
```

### Handling Numerical Stability Issues

If you encounter NaN values or training instability:

```bash
# For emergency recovery from NaN issues
./optimize_nan.sh

# Alternative emergency training script with extra stability settings
./emergency_training.sh
```

## Model-Specific Guidelines

### MLA Model

For the MLA (Multi-head Latent Attention) model:
- Use `--grad_clip 1.0` to maintain stability
- Start with BF16 precision before enabling FP8
- For H100/H200 GPUs, you can enable FP8 precision with `--use_fp8`
- Refer to `docs/numerical_stability.md` for detailed guidance

### LLaDA Model

For the LLaDA (Large Language Diffusion with mAsking) model:
- Requires special data preparation with masking
- Different inference approach using diffusion-based sampling
- See `docs/llada.md` for implementation details

## File Structure Overview

- `models/`: Model architecture implementations
  - `blocks/`: Basic building blocks (attention, MLP, normalization)
  - `deepseek/`: DeepSeek model variants
  - `llada/`: LLaDA model implementation
  - `models/`: High-level model classes including MLA
- `train/`: Training utilities
- `data/`: Data loading implementations
- `optimization/`: GPU optimization modules
- `docs/`: Implementation documentation

## Testing

There are no formal test scripts, but you can validate model implementations using small run configurations before full training.

## Memories and Current Focus

- We are currently working on mla_model and related blocks.
- If you want to run the project, use the virtualenv 5090 by running pyenv activate 5090
- Avoid creating new files just for testing purpose.

