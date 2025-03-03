# LLaDA: Large Language Diffusion with mAsking

This repository contains an implementation of the LLaDA (Large Language Diffusion with mAsking) model based on the paper [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992). This implementation combines the diffusion-based approach of LLaDA with a Mixture of Experts (MoE) architecture for enhanced performance.

## Key Features

- **Diffusion-based Language Model**: Unlike autoregressive models (like GPT), LLaDA uses a diffusion process with forward masking and reverse demasking
- **Mixture of Experts**: Utilizes a sparse MoE architecture where each token is processed by only a subset of expert networks
- **Bidirectional Attention**: No causal masking, allowing for more powerful representations
- **Variable Masking Ratios**: Uses masking ratios from 0-100% during training, making it a true generative model
- **Efficient Generation**: Semi-autoregressive generation that can be faster than autoregressive models

## Model Architecture

The LLaDA model combines several powerful ideas:

1. **Transformer Encoder with MoE**: Each layer contains bidirectional self-attention and MoE feed-forward networks
2. **Router**: Determines which experts should process each token
3. **Diffusion Process**: Uses forward masking (adding noise) and reverse demasking (removing noise) for generation
4. **Specialized Masking**: Special [MASK] token (ID 126336) used to mask tokens during diffusion

## Getting Started

### Installation

```bash
git clone https://github.com/yourusername/llada.git
cd llada
pip install -e .
```

### Training

To train a LLaDA model:

```bash
python train.py --mode pretrain \
                --n_layer 12 \
                --n_head 12 \
                --n_embd 768 \
                --num_experts 8 \
                --k 2 \
                --batch_size 32 \
                --learning_rate 4e-4 \
                --output_dir ./checkpoints
```

For supervised fine-tuning:

```bash
python train.py --mode finetune \
                --checkpoint ./checkpoints/llada_final.pt \
                --learning_rate 2.5e-5 \
                --batch_size 16 \
                --max_epochs 3
```

### Using with train_moe.py

This implementation also includes adapter classes to use LLaDA with the `train_moe.py` training script. There are two options:

#### Option 1: Using the adapter

The `llada/adapter.py` file provides compatibility classes to use LLaDA with `train_moe.py`:

```bash
# First make sure to run the adapter script
python train_llada_moe.py --n_layer 12 \
                         --n_head 12 \
                         --n_embd 768 \
                         --num_experts 8 \
                         --k 2 \
                         --batch_size 32 \
                         --learning_rate 4e-4 \
                         --output_dir ./out_llada
```

This will provide instructions on how to integrate LLaDA with `train_moe.py`.

#### Option 2: Using the standalone training script

Alternatively, use the standalone script that adapts `train_moe.py` directly for LLaDA:

```bash
python train_llada_with_moe.py --n_layer 12 \
                              --n_head 12 \
                              --n_embd 768 \
                              --num_experts 8 \
                              --k 2 \
                              --batch_size 32 \
                              --lr 4e-4 \
                              --out_dir ./out_llada
```

This script integrates all the necessary components from `train_moe.py` specifically for training LLaDA models. It supports both single-GPU and distributed training through PyTorch's distributed data parallel (DDP).

### Generation

To generate text using a trained model:

```bash
python generate.py --checkpoint ./checkpoints/llada_final.pt \
                  --prompt "Once upon a time" \
                  --length 256 \
                  --block_length 32 \
                  --remasking low_confidence
```

## Parameter Explanation

- **block_size**: Maximum sequence length
- **vocab_size**: Size of the vocabulary
- **n_layer**: Number of transformer layers
- **n_head**: Number of attention heads
- **n_embd**: Embedding dimension
- **num_experts**: Number of experts in the MoE layer
- **k**: Number of experts to route each token to
- **mask_token_id**: Special token ID for [MASK]
- **remasking**: Strategy for demasking tokens during generation ('low_confidence' or 'random')

## Generation Process

LLaDA generates text through an iterative diffusion process:

1. Initialize with fully masked sequence after the prompt
2. For a set number of steps, the model predicts the unmasked tokens
3. Based on the confidence score (or randomly), unmask a portion of tokens
4. Continue until all tokens are unmasked

### Remasking Strategies

- **low_confidence**: Unmask tokens with highest prediction confidence first
- **random**: Randomly select which tokens to unmask

## Supervised Fine-tuning

For SFT, a special forward pass is used:
- The prompt part is kept clean (not masked)
- Only the response part undergoes the diffusion process
- The loss is calculated only on masked tokens in the response

## Performance Tips

1. **Block Length**: Smaller blocks (32-64) work better for shorter generations, larger blocks (128-256) for longer ones
2. **Steps**: Equal to or more than generation length for best quality
3. **Temperature**: 0 for deterministic generation, >0 for diversity
4. **Remasking**: 'low_confidence' typically gives better quality, 'random' can be more diverse

## Citation

If you use this code, please cite the original LLaDA paper:

```bibtex
@article{llada2024,
  title={Large Language Diffusion Models},
  author={Ding, Ning and Jiao, Wenxiang and Qiu, Chen and Liu, Zhifang and Weng, Yikang and You, Haida and Bu, Leyang and Chen, Yi and Qian, Zejian and Zheng, HaoZhe and Qin, Chao and Gong, Zheng and Zhou, Bowen and Yang, Yi and Cao, Yixin and Zhou, Li and Lou, Jian-Guang and Zhou, Bowen},
  journal={arXiv preprint arXiv:2502.09992},
  year={2024}
}
```

## License

This project is released under the MIT License. 