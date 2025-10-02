# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT RULES

Don't try to go around issues if can fix them. Don't by pass errors, fix bugs.
When it's possible, try to factorize code to make it more readable and maintainable.

## Repository Overview

GPToughts is a framework for training and fine-tuning Large Language Models (LLMs), with a focus on GPU optimizations for hobby-scale training. The repository supports multiple model architectures including GPT-style autoregressive models, DeepSeek models, and LLaDA (Large Language Diffusion with mAsking) models which use a diffusion-based approach.

## Project Architecture

The codebase is organized into several key components:

1. **Model Architectures**:
   - Standard autoregressive models (GPT-style)
   - DeepSeek models with various adapters and MTP (Multi-Token Prediction)
   - LLaDA models (diffusion-based approach)
   - MLA (Multi-head Latent Attention) models
   - MoE (Mixture of Experts) variants
   - SLM, NSA, HRM models for specialized research

2. **Training Infrastructure**:
   - PyTorch Lightning modules (`train/lightning_module.py`)
   - Multiple data loader implementations (`data/data_loader_*.py`)
   - Memory optimization utilities (`train/memory_utils.py`)
   - Progressive training system (`train_progressive.py`)

3. **GPU Optimizations**:
   - FP8 precision support for H100/H200 GPUs
   - Memory usage optimizations via `optimization/` module
   - CUDA-specific performance enhancements
   - GaLore optimizer for memory-efficient training
   - AdEMAMix optimizer with adaptive momentum

## Development Environment

```bash
# Use the recommended Python environment
pyenv activate 5090

# Install dependencies
pip install -r requirements.txt

# Set up Hugging Face token (needed for accessing some models)
export HF_TOKEN=your_huggingface_token
```

## Key Commands

### Basic Training

```bash
# Basic training command with all available model types
python run_train.py --model_type [model_type] --size [size] --batch_size [batch_size] --block_size [block_size]

# Available model types: deepseek, llada, sedd, gpt, mla, mla_selective, parscale_mla, mla_llada, mdm, moe_mla, slm, nsa, hrm
# Available sizes: small, medium, large, xl

# Example: Train MLA model with specific configuration
python run_train.py --model_type mla --size small --batch_size 8 --block_size 2048
```

### Model-Specific Training Scripts

```bash
# MLA models with different optimizations
./train_mla_optimized.sh          # MLA with all optimizations
./train_mla_galore.sh             # MLA with GaLore optimizer
./train_mla_selective.sh          # MLA with selective attention
./train_parscale_mla.sh           # MLA with ParScale normalization

# DeepSeek models
./train_deepseek_mtp.sh           # DeepSeek with Multi-Token Prediction

# LLaDA models
./train_llada_optimized.sh        # LLaDA with optimizations
./train_llada_bd3_fixed.sh        # LLaDA with BD3 fixes

# Mixture of Experts
./train_moe_mla_galore2.sh        # MoE MLA with GaLore

# Specialized models
./train_slm.sh                    # SLM training
./train_nsa.sh                    # NSA training
./train_hrm.sh                    # HRM training
```

### Progressive Training (Recommended)

Progressive training starts with shorter sequences and gradually increases length, improving stability:

```bash
# Easy progressive training with default settings
./train_progressive_easy.sh [model_type] [size] [batch_size] [dataset] [output_dir]

# Examples:
./train_progressive_easy.sh mla small 8
./train_progressive_easy.sh gpt medium 4
./train_progressive_easy.sh llada small 16

# Full progressive training with custom settings
python train_progressive.py --model_type mla --size small \
    --progressive_block_sizes 256,512,1024,2048,4096 \
    --progressive_epochs_per_stage 5 \
    --use_position_interpolation \
    --max_epochs 25
```

### Model Adaptation for Different Sequence Lengths

```bash
# Adapt a model trained on one sequence length for use with longer sequences
python example_load_with_new_block_size.py \
    --checkpoint_path out/checkpoints/model-1024.ckpt \
    --original_block_size 1024 \
    --new_block_size 2048 \
    --model_type mla \
    --size small \
    --use_position_interpolation
```

### GPU Optimization System

```bash
# View all optimization options
./optimize.sh --help

# Enable all optimizations
./optimize.sh --all

# Specific optimizations
./optimize.sh --memory --cuda --fp8

# Automatic batch size optimization
./optimize.sh --auto-batch

# Enable compilation and advanced optimizations
./optimize.sh --compile --attention --all

# Recovery from numerical instability
./optimize.sh --recover-nan

# Launch training with optimizations
./optimize.sh --all --train --model mla --size medium
```

## Data Loading System

The repository includes multiple data loader implementations optimized for different scenarios:

- `data_loader_dynamic.py`: Dynamic batching for variable sequence lengths
- `data_loader_packed.py`: Packed sequences for efficiency
- `data_loader_concatenated.py`: Concatenated documents
- `data_loader_llada.py`: Specialized for LLaDA diffusion models
- `data_loader_legacy.py`: Original implementation for compatibility

## Model-Specific Guidelines

### MLA (Multi-head Latent Attention)

```bash
# Recommended training approach
python run_train.py --model_type mla --size small \
    --grad_clip 1.0 \
    --learning_rate 3e-4 \
    --precision bf16-mixed \
    --optimizer galore
```

Key considerations:
- Use `--grad_clip 1.0` for stability
- Start with BF16 precision before enabling FP8
- Keep MLA parameters in higher precision even when using FP8
- Refer to `docs/numerical_stability.md` for detailed guidance

### LLaDA (Large Language Diffusion with mAsking)

```bash
# LLaDA requires special masking data preparation
python run_train.py --model_type llada --size small \
    --dataloader_type llada \
    --diffusion_steps 1000
```

Key considerations:
- Requires diffusion-based data preparation with masking
- Different inference approach using iterative denoising
- See `docs/llada.md` for implementation details

### DeepSeek with Multi-Token Prediction

```bash
# DeepSeek MTP for improved efficiency
python run_train.py --model_type deepseek --size medium \
    --use_mtp \
    --mtp_loss_weight 0.3
```

### Mixture of Experts (MoE)

```bash
# MoE training with expert routing
python run_train.py --model_type moe_mla --size large \
    --num_experts 8 \
    --top_k_experts 2 \
    --expert_capacity_factor 1.0
```

## Critical Architecture Components

### Model Block Structure

Models are constructed from reusable blocks in `models/blocks/`:
- Attention mechanisms (standard, MLA, selective)
- MLP layers with various activations
- Normalization layers (LayerNorm, RMSNorm, ParScale)
- Embedding and positional encoding

### Training Loop Architecture

The training system uses PyTorch Lightning with:
- `train/lightning_module.py`: Main Lightning module
- `train/train_utils.py`: Checkpoint management and utilities
- `train/memory_utils.py`: Memory optimization helpers

### Optimization Pipeline

The `optimization/` module provides:
- `cuda_optim.py`: CUDA-specific optimizations
- `fp8_optim.py`: FP8 precision optimizations  
- `memory_optim.py`: Memory usage optimizations
- `training_optim.py`: Training workflow optimizations

## Numerical Stability Management

The framework includes comprehensive stability features:

1. **Gradient Clipping**: Always use `--grad_clip 1.0` for MLA models
2. **Precision Hierarchy**: Keep critical parameters in higher precision
3. **Progressive Training**: Start with shorter sequences
4. **Recovery Tools**: Automatic NaN detection and recovery
5. **Monitoring**: Extensive logging of numerical stability metrics

Refer to `docs/numerical_stability.md` for detailed stability guidelines.

## Testing and Validation

```bash
# Quick validation with small configuration
python run_train.py --model_type [model] --size small \
    --batch_size 1 --block_size 128 --max_steps 10

# Test specific model parameters
python test_slm_params.py  # For SLM model validation
python test_llada_config.py  # For LLaDA configuration testing
```

## Documentation Resources

- `docs/mla_doc.md`: Detailed MLA architecture documentation
- `docs/llada.md`: LLaDA implementation guide  
- `docs/numerical_stability.md`: Stability best practices
- `docs/parscale_mla.md`: ParScale normalization usage
- `docs/selective_attention.md`: Selective attention mechanisms
- `OPTIMIZATION_README.md`: GPU optimization system overview

## Current Development Focus

- MLA model architecture refinements and stability improvements
- Advanced MoE implementations with improved routing
- FP8 precision optimization for H100/H200 GPUs
- Progressive training system enhancements
- Numerical stability improvements across all model types

## Environment Setup

Always use the recommended Python environment:
```bash
pyenv activate 5090
```

Avoid creating new files for testing purposes - use existing validation scripts with small configurations instead.

