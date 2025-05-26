# ParScale-MLA Usage Guide

## Quick Start

ParScale-MLA can now be trained using the standard `run_train.py` script with PyTorch Lightning.

### Stage 1: Train Base MLA Model

```bash
python run_train.py \
    --model_type mla \
    --size small \
    --batch_size 8 \
    --block_size 2048 \
    --output_dir out_mla_base \
    --max_iters 10000 \
    --use_lightning
```

### Stage 2: Train ParScale Components

```bash
python run_train.py \
    --model_type parscale_mla \
    --size small \
    --batch_size 8 \
    --block_size 2048 \
    --output_dir out_parscale \
    --max_iters 2000 \
    --parallel_streams 8 \
    --training_stage 2 \
    --base_checkpoint out_mla_base/last.ckpt \
    --freeze_base_in_stage2 \
    --use_lightning
```

## Using the Convenience Script

The `train_parscale_mla.sh` script simplifies training:

```bash
# Stage 1: Train base model
./train_parscale_mla.sh small 8 2048 out_base 8 1

# Stage 2: Train ParScale components  
./train_parscale_mla.sh small 8 2048 out_parscale 8 2 out_base/last.ckpt
```

Parameters:
1. Model size (small/medium/large/xl)
2. Batch size
3. Block size (sequence length)
4. Output directory
5. Number of parallel streams
6. Training stage (1 or 2)
7. Base checkpoint path (for stage 2)

## Command Line Options

### ParScale-Specific Parameters

- `--parallel_streams`: Number of parallel streams (default: 8)
- `--prefix_length`: Length of input-space prefixes (default: 48)
- `--latent_prefix_length`: Length of latent-space prefixes (default: 16)
- `--aggregator_epsilon`: Label smoothing for aggregation (default: 0.1)
- `--diversity_weight`: Weight for diversity regularization (default: 0.1)
- `--use_dynamic_inference`: Enable complexity-based stream allocation
- `--complexity_threshold`: Threshold for full stream activation (default: 0.5)
- `--training_stage`: Training stage 1 (base) or 2 (parscale)
- `--freeze_base_in_stage2`: Freeze base model in stage 2
- `--base_checkpoint`: Base model checkpoint for stage 2

## Monitoring Training

### Standard Metrics
- `train/loss`: Training loss
- `val/loss`: Validation loss
- `val/perplexity`: Validation perplexity
- `tokens_per_sec_step`: Training throughput

### ParScale-Specific Metrics
- `parscale/input_similarity`: Similarity between input prefixes
- `parscale/layer_similarity`: Average similarity across layers
- `parscale/effective_streams`: Effective number of diverse streams

## Example Workflow

1. **Train base MLA model** (98% of training budget):
   ```bash
   ./train_parscale_mla.sh medium 16 4096 out_mla_base 8 1
   ```

2. **Monitor training** with TensorBoard or WandB:
   ```bash
   tensorboard --logdir out_mla_base/logs
   ```

3. **Train ParScale components** (2% of training budget):
   ```bash
   ./train_parscale_mla.sh medium 16 4096 out_parscale 8 2 out_mla_base/last.ckpt
   ```

4. **Evaluate improvements**:
   - Check diversity metrics to ensure streams are specialized
   - Compare validation perplexity with base model
   - Test on reasoning tasks for expected 30-40% improvement

## Tips

1. **Stream Diversity**: Monitor `parscale/effective_streams`. Should be close to `parallel_streams`.

2. **Prevent Collapse**: If streams converge (low diversity), increase:
   - `aggregator_epsilon` to 0.15-0.2
   - `diversity_weight` to 0.2-0.3

3. **Memory Usage**: ParScale adds only ~4.5% memory overhead with 8 streams.

4. **Inference Speed**: Use dynamic inference to reduce computation on simple queries:
   ```python
   model.dynamic_inference(input_ids, complexity_threshold=0.3)
   ```

## Integration with Existing Code

ParScale-MLA is fully integrated with the training infrastructure:
- Works with all optimizers (AdamW recommended)
- Supports gradient accumulation
- Compatible with mixed precision training
- Integrates with wandb/tensorboard logging
- Supports distributed training

## Troubleshooting

### Low Stream Diversity
- Increase `diversity_weight`
- Check prefix initialization
- Ensure base model is well-trained

### OOM Errors
- Reduce `parallel_streams` (try 4)
- Enable gradient checkpointing
- Use smaller batch size

### Slow Convergence in Stage 2
- Increase learning rate to 5e-4
- Ensure base model has converged
- Check that base weights are frozen