# Numerical Stability Guide for MLA Model Training

This document provides guidance for maintaining numerical stability when training models using the MLA (Multi-head Latent Attention) architecture with sparse MoE (Mixture of Experts) components. It also includes information about critical bugs that were fixed in the implementation.

## Numerical Stability Challenges

The MLA-Model architecture presents several numerical stability challenges:

1. **Routing in MoE layers** - Routing can become unstable, especially at the beginning of training.
2. **FP8 precision** - Using FP8 can lead to increased instability in the early layers.
3. **Reduced dimensionality in MLA** - Low-rank projections in MLA can amplify numerical instabilities.
4. **Extreme sparsity** - With a large number of experts and few experts activated per token, the signal can degrade.
5. **Tensor shape issues** - Some operations like `view()` require contiguous memory layouts and can fail with non-contiguous tensors.
6. **Dimension mismatches** - The MLA components have different dimension calculations that can cause runtime errors.

## Recommended Configurations

### Precision Configuration

- **Critical MLA parameters** - Keep MLA parameters in FP16/BF16 even when using FP8 for the rest of the model.
  ```bash
  # Enable FP8 but keep MLA in FP16
  ./train_simplified.sh [size] [batch_size] [block_size] [output_dir] 1
  # Don't use --fp8_mla_params unless you're sure of stability
  ```

- **Precision by model size**
  - Small/Medium: BF16 is generally stable
  - Large/XL: Start with BF16 without FP8, then enable FP8 after initial convergence

### Hyperparameters

- **Learning Rate**
  - Start with lower LRs (1e-4 to 3e-4)
  - Increase warmup period (1000-2000 steps)
  - Cyclic LR can help avoid unstable local minima

- **Batch Size**
  - Smaller batch sizes are more stable (8-16)
  - Increase gradient_accumulation_steps rather than batch_size
  - For large models, limit to 2-4 sequences per GPU

- **Gradient Clipping**
  - Essential for MLA-Model: `--grad_clip 1.0`
  - For extreme problems, reduce to 0.5

### Architectural

- **Dense & MoE Layers**
  - First layers (4-8) should remain dense (`min_moe_layer=4` by default)
  - These dense layers stabilize representations before MoE routing

- **Routing Parameters**
  - If you encounter NaNs, reduce `router_z_loss_coef` (0.001 → 0.0001)
  - Increase the number of experts activated per token for more stability

## Tensor Operations and Memory Layout

### Common Issues and Solutions

1. **View vs Reshape Operations**
   - `view()` requires contiguous memory and will fail if tensor is non-contiguous
   - Use `reshape()` as a safer alternative when tensor layout is uncertain
   - For operations after transposing or permuting, always use `reshape()`

2. **Tensor Contiguity**
   - After operations like `transpose()`, `permute()`, or `select()`, tensors become non-contiguous
   - Call `.contiguous()` before using `view()` or use `reshape()` directly
   - Example: `x = x.transpose(1, 2).contiguous().view(new_shape)`

3. **Einsum Operations**
   - When using complex einsum patterns, check tensor shapes and memory layout
   - Consider using explicit reshape operations for better control

## Critical Fixes for MLA-Model Implementation

Several critical issues have been fixed in the MLA-Model implementation to prevent runtime errors and improve stability.

### 1. Dimension Mismatch in Expert Adapters

**Problem**: 
- The code had a critical dimension mismatch between `SharedExpertMLP` and `ExpertGroup` classes
- `SharedExpertMLP` defined `adapt_dim = hidden_dim // 16` where `hidden_dim = 4 * config.n_embd`
- `ExpertGroup` defined `adapt_dim = hidden_dim // 16` where `hidden_dim = 2 * config.n_embd`
- This caused a 2x difference that resulted in matrix multiplication errors:
  ```
  RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x192 and 96x96)
  ```

**Solution**:
- Modified `ExpertGroup` to use the same `adapt_dim` as `SharedExpertMLP`
- Ensured the `SharedExpertMLP` instance is created first, then its `adapt_dim` is reused

### 2. Inconsistent Return Signatures

**Problem**:
- `MLABlock` returns a single tensor `x`
- `MLAModelBlock` returns a tuple `(x, router_loss)`
- This caused unpacking errors when processing blocks

**Solution**:
- Modified `MLAModel.forward()` to handle both return types
- Added type checking to correctly extract values from block outputs

### 3. Tensor Shape Mismatch in MoE Layer

**Problem**:
- When assigning expert outputs back to the combined output tensor, there were shape mismatches
- The tensor shape error: `shape [39, 3072] cannot be broadcast to indexing result of shape [39, 768]`

**Solution**:
- Added dynamic reshaping for expert outputs to match input dimensions
- Implemented padding/truncation where needed to ensure compatibility

### 4. Data Type Mismatch

**Problem**:
- Tensors with different data types were causing errors when combined
- The error: `Index put requires the source and destination dtypes match, got Float for the destination and BFloat16 for the source`

**Solution**:
- Added automatic data type conversion to ensure consistent types
- Used `.to(dtype=target_dtype)` to convert tensors before operations

### 5. FP8 Precision Issues

**Problem**:
- When using FP8 precision, we encountered promotion errors:
  ```
  RuntimeError: Promotion for Float8 Types is not supported, attempted to promote Float and Float8_e4m3fn
  ```

**Solution**:
- FP8 is only supported on specific hardware (H100/H200/Ada Lovelace GPUs)
- Added a parameter to disable FP8 for testing/development on other hardware
- Implemented dtype-aware conversion in normalization layers

## Testing Fixed Implementation

A test script `test_mla.py` has been created to validate the fixes:

```bash
# Run with default settings
python test_mla.py

# Run with specific parameters
python test_mla.py --size small --batch_size 2 --block_size 128 --fp8_params False
```

The script tests both forward pass and text generation to ensure all components work correctly.

## Detection and Recovery Techniques

### Detection

MLA-Model includes several mechanisms to detect numerical problems:

1. Automatic NaN/Inf checking in each forward pass
2. Separate logging of routing losses for tracking
3. Clipping of extreme activations

### Recovery

If you encounter instabilities:

1. **Automatic recovery** - The model will attempt to clip extreme values
2. **Checkpoint recovery** - Resume from a stable checkpoint with reduced LR
3. **Emergency script** - Use `emergency_training.sh` (based on `optimize_nan.sh`) to apply additional constraints

```bash
# Recover from an unstable session
./emergency_training.sh [checkpoint_dir]
```

## Recommended Optimizations by GPU

| GPU | Precision | Optimizations |
|-----|-----------|---------------|
| H100/H200 | BF16→FP8 | `--use_fp8 --optimize_attention --preallocate_memory` |
| A100 | BF16 | `--optimize_attention --preallocate_memory` |
| RTX 4090/RTX 3090 | BF16 | `--optimize_attention` |
| Other GPUs | FP16 | No special optimizations |

## Example Configuration for Maximum Stability

For a particularly difficult model to train, use:

```bash
python run_train.py \
  --model_type mla \
  --size medium \
  --batch_size 4 \
  --grad_clip 0.5 \
  --learning_rate 1e-4 \
  --warmup_iters 2000 \
  --min_lr 1e-5 \
  --dropout 0.1 \
  --router_z_loss_coef 0.0001
```

## For DeepSeek-Like MLA-Model

To reproduce a DeepSeek-like architecture with MLA, use:

```bash
python run_train.py \
  --model_type mla \
  --size medium \
  --batch_size 8 \
  --block_size 4096 \
  --optimizer_type lion \
  --grad_clip 1.0
```