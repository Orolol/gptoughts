# Complete Guide to Building a LLaDA Model (Large Language Diffusion with mAsking)

This document provides comprehensive instructions for implementing a LLaDA model from scratch, based on the paper "Large Language Diffusion Models" (https://arxiv.org/abs/2502.09992).

## 1. Theoretical Overview

LLaDA fundamentally differs from autoregressive language models (like GPT) by using a diffusion-based approach with masked tokens. Key points:

- LLaDA uses a **diffusion process** with forward masking and reverse demasking
- It trains a **mask predictor** (Transformer encoder) that predicts original tokens from masked inputs
- The training objective forms an **upper bound on negative log-likelihood**
- It can achieve comparable performance to autoregressive models while having bidirectional attention

Unlike BERT, which uses a fixed masking ratio (~15%), LLaDA uses variable masking ratios from 0-100%, making it a true generative model with theoretical guarantees.

## 2. Model Architecture

LLaDA uses a standard Transformer encoder architecture with these modifications:

```python
# Starting with a standard Transformer encoder (similar to BERT)
# Key differences:
# 1. No causal masking in self-attention (bidirectional attention)
# 2. Reserve a special token for [MASK] (usually token ID 126336)
# 3. Use the same architecture as GPT/LLaMA but remove causal masking

class LLaDAModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        mask_token_id=126336,
        **kwargs
    ):
        super().__init__()
        self.mask_token_id = mask_token_id
        
        # Standard transformer components
        self.embeddings = TransformerEmbeddings(vocab_size, hidden_size)
        self.encoder = TransformerEncoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            **kwargs
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def forward(self, input_ids, attention_mask=None):
        # Normal transformer forward pass, but NO causal masking
        hidden_states = self.embeddings(input_ids)
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
        )
        logits = self.lm_head(encoder_outputs)
        return logits
```

## 3. Pre-training Implementation

### Forward Process (Masking)

```python
def forward_process(input_ids, eps=1e-3):
    """Apply random masking with varying ratios."""
    batch_size, seq_len = input_ids.shape
    
    # Sample random masking ratios between eps and 1-eps
    t = torch.rand(batch_size, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, seq_len)
    
    # Apply masking randomly according to p_mask
    masked_indices = torch.rand((batch_size, seq_len), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, 126336, input_ids)  # 126336 is [MASK] token ID
    
    return noisy_batch, masked_indices, p_mask
```

### Training Loop

```python
def train_llada(model, tokenizer, dataset, optimizer, num_epochs, batch_size=16):
    """Train a LLaDA model."""
    model.train()
    
    for epoch in range(num_epochs):
        for batch in DataLoader(dataset, batch_size=batch_size):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"]
            
            # With 1% probability, use variable sequence length (helps model generalization)
            if torch.rand(1) < 0.01:
                random_length = torch.randint(1, input_ids.shape[1] + 1, (1,))
                input_ids = input_ids[:, :random_length]
            
            # Forward process: apply masking
            noisy_batch, masked_indices, p_mask = forward_process(input_ids)
            
            # Model forward pass
            logits = model(noisy_batch).logits
            
            # Calculate loss only on masked tokens
            token_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1))[masked_indices.view(-1)], 
                input_ids.view(-1)[masked_indices.view(-1)], 
                reduction='none'
            ) / p_mask.view(-1)[masked_indices.view(-1)]
            
            # Normalize loss
            loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```

### Learning Rate Schedule

LLaDA uses a "Warmup-Stable-Decay" learning rate scheduler:
1. Linear warmup from 0 to 4e-4 for first 2000 iterations
2. Constant 4e-4 until 1.2T tokens processed
3. Decay to 1e-4 and hold for 0.8T tokens
4. Linear decay from 1e-4 to 1e-5 for last 0.3T tokens

## 4. Supervised Fine-Tuning (SFT)

SFT requires preprocessing data to handle prompts and responses correctly. Key differences from pre-training:

```python
def sft_forward_step(model, batch):
    """Supervised fine-tuning step for LLaDA."""
    input_ids, prompt_lengths = batch["input_ids"], batch["prompt_lengths"]
    
    # Apply forward process (masking) to entire sequence
    noisy_batch, masked_indices, p_mask = forward_process(input_ids)
    
    # Do not add noise to the prompt part - keep it clean
    for i in range(input_ids.shape[0]):
        prompt_mask = torch.arange(noisy_batch.shape[1], device=noisy_batch.device) < prompt_lengths[i]
        noisy_batch[i, prompt_mask] = input_ids[i, prompt_mask]
    
    # Calculate the answer length (including padded <EOS> tokens)
    answer_lengths = input_ids.shape[1] - prompt_lengths
    
    # Get model predictions
    logits = model(noisy_batch).logits
    
    # Only calculate loss on masked tokens in the response
    response_mask = masked_indices & ~torch.cat([
        torch.arange(noisy_batch.shape[1], device=noisy_batch.device) < prompt_lengths[i, None]
        for i in range(input_ids.shape[0])
    ], dim=0)
    
    # Compute loss on masked response tokens only
    token_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1))[response_mask.view(-1)],
        input_ids.view(-1)[response_mask.view(-1)],
        reduction='none'
    ) / p_mask.view(-1)[response_mask.view(-1)]
    
    # Normalize by answer length
    ce_loss = torch.sum(token_loss / answer_lengths.view(-1)[response_mask.view(-1)]) / input_ids.shape[0]
    
    return ce_loss
```

## 5. Inference and Sampling

### Basic Sampling Function

```python
@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.0, remasking='low_confidence', mask_id=126336):
    """Generate text using LLaDA diffusion process."""
    device = prompt.device
    
    # Initialize with fully masked sequence
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    # Track prompt indices to avoid modifying them
    prompt_index = (x != mask_id)
    
    # Calculate blocks for semi-autoregressive generation
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    # Adjust steps per block
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    
    # Generate each block
    for block_idx in range(num_blocks):
        # Get mask indices for current block
        start_idx = prompt.shape[1] + block_idx * block_length
        end_idx = prompt.shape[1] + (block_idx + 1) * block_length
        block_mask_index = (x[:, start_idx:end_idx] == mask_id)
        
        # Calculate tokens to unmask at each step
        mask_count = block_mask_index.sum(dim=1, keepdim=True)
        tokens_per_step = torch.div(mask_count, steps_per_block, rounding_mode='floor')
        remainder = mask_count % steps_per_block
        
        num_transfer_tokens = torch.full((1, steps_per_block), tokens_per_step, device=device)
        num_transfer_tokens[:, :remainder] += 1
        
        # Iteratively unmask tokens
        for step in range(steps_per_block):
            # Get current mask indices
            mask_index = (x == mask_id)
            
            # Get model predictions
            logits = model(x).logits
            
            # Apply optional temperature sampling
            if temperature > 0:
                # Add Gumbel noise for stochastic sampling
                noise = torch.rand_like(logits)
                gumbel_noise = -torch.log(-torch.log(noise + 1e-10) + 1e-10) * temperature
                logits = logits + gumbel_noise
            
            # Get token predictions
            x0 = torch.argmax(logits, dim=-1)
            
            # Calculate confidence scores for low-confidence remasking
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == 'random':
                x0_p = torch.rand_like(x0, dtype=torch.float)
            else:
                raise ValueError(f"Unknown remasking strategy: {remasking}")
            
            # Don't consider future blocks for unmasking
            x0_p[:, end_idx:] = float('-inf')
            
            # Update predictions only for masked tokens
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.tensor(float('-inf'), device=device))
            
            # Select tokens to unmask based on confidence
            transfer_index = torch.zeros_like(x, dtype=torch.bool)
            for i in range(x.shape[0]):
                # Use topk to find indices of highest confidence predictions
                if remasking == 'low_confidence':
                    _, indices = torch.topk(confidence[i], k=num_transfer_tokens[i, step].item())
                else:  # random remasking
                    # Randomly select indices among masked tokens
                    masked_positions = torch.where(mask_index[i])[0]
                    perm = torch.randperm(len(masked_positions), device=device)
                    indices = masked_positions[perm[:num_transfer_tokens[i, step].item()]]
                
                transfer_index[i, indices] = True
            
            # Update x with unmasked tokens
            x[transfer_index] = x0[transfer_index]
    
    return x
```

### Evaluating Log-Likelihood

For tasks requiring likelihood estimation:

```python
@torch.no_grad()
def get_log_likelihood(model, prompt, answer, mc_num=128, batch_size=16, mask_id=126336):
    """Estimate log-likelihood of answer given prompt using Monte Carlo."""
    device = prompt.device
    seq = torch.cat([prompt, answer])[None, :].repeat(batch_size, 1).to(device)
    prompt_index = torch.arange(seq.shape[1], device=device) < len(prompt)
    
    losses = []
    for _ in range(mc_num // batch_size):
        # Apply random masking with uniform sampling of mask count
        l = seq.shape[1] - len(prompt)  # answer length
        k = torch.randint(1, l + 1, (batch_size,), device=device)  # number of masks per batch
        
        mask_index = torch.zeros_like(seq, dtype=torch.bool)
        for i in range(batch_size):
            # Randomly select k[i] positions in the answer to mask
            answer_indices = torch.arange(len(prompt), seq.shape[1], device=device)
            perm = torch.randperm(l, device=device)
            to_mask = answer_indices[perm[:k[i]]]
            mask_index[i, to_mask] = True
        
        # Apply masking
        x = seq.clone()
        x[mask_index] = mask_id
        
        # Get model predictions
        logits = model(x).logits
        
        # Calculate loss only on masked tokens
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1))[mask_index.view(-1)],
            seq.view(-1)[mask_index.view(-1)],
            reduction='none'
        )
        
        # Adjust by mask ratio k/l
        mask_ratio = k.float() / l
        adjusted_loss = loss.sum() / mask_ratio.sum()
        
        losses.append(adjusted_loss.item())
    
    # Return negative log-likelihood
    return -sum(losses) / len(losses)
```

## 6. Data Preparation and Tokenization

LLaDA uses the same tokenizer and data preparation as standard LLMs:

1. **Pre-training data**: Standard web text, books, code, and multilingual corpora
2. **SFT data**: Instruction-response pairs, with special handling for multi-turn dialogues

### SFT Data Format

For Supervised Fine-Tuning, data should be preprocessed as:

```
input_ids:
<BOS><start_id>user<end_id>\nQuestion?<eot_id><start_id>assistant<end_id>\nAnswer.<EOS><EOS>...<EOS>
prompt_lengths: [length_of_prompt_tokens]
```

- The EOS tokens are padding to ensure equal length in a batch
- These are treated as part of the response during training
- During inference, EOS tokens help the model control response length

## 7. Remasking Strategies

During generation, LLaDA offers different remasking strategies:

1. **Random remasking**: Randomly select tokens to remask (simpler)
2. **Low-confidence remasking**: Remask tokens with lowest prediction confidence (better quality)
3. **Semi-autoregressive remasking**: Generate text in blocks from left to right (helps with long responses)

For LLaDA-Base: Use low-confidence remasking
For LLaDA-Instruct: Use low-confidence with semi-autoregressive for best results

## 8. Hyperparameters and Training Settings

### Architecture (8B model)
- Layers: 32
- Model dimension: 4096
- Attention heads: 32
- Key/Value heads: 32
- FFN dimension: 12288
- Vocabulary size: 126,464

### Pre-training
- Tokens: 2.3 trillion
- Learning rate: Warmup to 4e-4, then 1e-4, final decay to 1e-5
- Weight decay: 0.1
- Batch size: 1280
- Sequence length: 4096
- Variable length training: 1% of batches

### SFT
- Pairs: 4.5 million
- Learning rate: 2.5e-5, decay to 2.5e-6
- Weight decay: 0.1
- Batch size: 256
- Training epochs: 3

### Inference
- Answer length: 256-1024 depending on task
- Sampling steps: Usually equal to answer length for best quality
- Block length: 32-512 depending on task and model type

## 9. Implementation Tips

1. **Start small**: Begin with a 100M parameter model on smaller datasets
2. **Checkpointing**: Save checkpoints regularly as training can occasionally become unstable
3. **Validation**: Monitor perplexity on a held-out validation set
4. **Inference optimization**: Experiment with fewer sampling steps and block sizes for faster inference
5. **Hardware requirements**: Full-scale training requires enterprise GPU clusters, but smaller models can be trained on fewer GPUs

## 10. Troubleshooting

- If training becomes unstable (NaN losses), reduce the learning rate
- If the model generates too many EOS tokens during inference, try random remasking instead of low-confidence
- For multi-turn conversations, use the semi-autoregressive approach with block_length=32

This completes the comprehensive guide for implementing a LLaDA model from scratch based on the paper and provided code.