"""
ParScale-MLA: Multi-head Latent Attention with Parallel Scaling.

This module implements ParScale (Parallel Scaling) for MLA models, enabling
O(log P) parameter scaling efficiency through P parallel streams with
learnable transformations and dynamic aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
import copy

from models.models.mla_model import MLAModel, MLAModelConfig, MLAModelBlock
from models.blocks.parscale import (
    LatentPrefixMLA, 
    DynamicAggregator, 
    ParallelMLACache,
    diversity_loss,
    ComplexityEstimator
)


class ParScaleMLAConfig(MLAModelConfig):
    """Configuration for ParScale-MLA model."""
    # ParScale specific parameters
    parallel_streams: int = 8  # Number of parallel streams (P)
    prefix_length: int = 48  # Length of input-space prefixes
    latent_prefix_length: int = 16  # Length of latent-space prefixes
    aggregator_epsilon: float = 0.1  # Label smoothing for aggregation
    diversity_weight: float = 0.1  # Weight for diversity regularization
    
    # Dynamic inference
    use_dynamic_inference: bool = True
    complexity_threshold: float = 0.5
    
    # Training strategy
    stage2_tokens_ratio: float = 0.02  # Use 2% of tokens for stage 2
    freeze_base_in_stage2: bool = True  # Freeze base model in stage 2
    
    # Optimization
    use_parallel_cache: bool = True
    cache_max_length: int = 8192


class ParScaleMLA(nn.Module):
    """
    ParScale-MLA: Implements parallel scaling for Multi-head Latent Attention models.
    
    This model creates P parallel streams with learnable prefix transformations,
    executes them in parallel through a shared MLA backbone, and dynamically
    aggregates the outputs.
    """
    
    def __init__(self, base_model: Optional[MLAModel] = None, config: Optional[ParScaleMLAConfig] = None):
        super().__init__()
        
        # Initialize from base model or config
        if base_model is not None:
            self.base_model = base_model
            self.config = ParScaleMLAConfig(**base_model.config.__dict__)
            # Update with ParScale specific configs if provided
            if config is not None:
                for key, value in config.__dict__.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
        elif config is not None:
            self.config = config
            # Create base model from config
            base_config = MLAModelConfig(**{
                k: v for k, v in config.__dict__.items() 
                if hasattr(MLAModelConfig, k)
            })
            self.base_model = MLAModel(base_config)
        else:
            raise ValueError("Either base_model or config must be provided")
        
        # ParScale components
        self.P = self.config.parallel_streams
        
        # Layer-wise prefix modules
        self.layer_prefixes = nn.ModuleDict({
            f'layer_{i}': LatentPrefixMLA(
                P=self.P,
                d_model=self.config.n_embd,
                d_latent=self.config.kv_lora_rank,  # Use KV latent dimension
                prefix_length=self.config.latent_prefix_length
            ) for i in range(self.config.n_layer)
        })
        
        # Global input prefixes
        self.global_input_prefixes = nn.Parameter(
            torch.randn(self.P, self.config.prefix_length, self.config.n_embd) * 0.02
        )
        
        # Dynamic aggregator
        self.aggregator = DynamicAggregator(
            d_model=self.config.n_embd,
            P=self.P,
            epsilon=self.config.aggregator_epsilon
        )
        
        # Optional complexity estimator for dynamic inference
        if self.config.use_dynamic_inference:
            self.complexity_estimator = ComplexityEstimator(
                d_model=self.config.n_embd,
                vocab_size=self.config.vocab_size
            )
        
        # Parallel caches for inference
        if self.config.use_parallel_cache:
            self.parallel_caches = {
                f'layer_{i}': ParallelMLACache(
                    P=self.P,
                    max_length=self.config.cache_max_length,
                    d_latent=self.config.kv_lora_rank,
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                ) for i in range(self.config.n_layer)
            }
        
        # Training stage tracker
        self.training_stage = 1  # Start with stage 1
        
        # Initialize ParScale parameters
        self._init_parscale_params()
    
    def _init_parscale_params(self):
        """Initialize ParScale-specific parameters."""
        # Orthogonalize global input prefixes
        with torch.no_grad():
            for i in range(1, self.P):
                for j in range(i):
                    # Gram-Schmidt orthogonalization
                    self.global_input_prefixes.data[i] -= (
                        torch.sum(self.global_input_prefixes.data[i] * self.global_input_prefixes.data[j]) * 
                        self.global_input_prefixes.data[j]
                    )
                self.global_input_prefixes.data[i] = F.normalize(
                    self.global_input_prefixes.data[i], dim=-1
                )
    
    def set_training_stage(self, stage: int):
        """
        Set training stage (1 or 2).
        
        Stage 1: Normal training of base model
        Stage 2: ParScale training with frozen base model
        """
        self.training_stage = stage
        
        if stage == 2 and self.config.freeze_base_in_stage2:
            # Freeze base model parameters
            for param in self.base_model.parameters():
                param.requires_grad = False
            
            # Ensure ParScale parameters are trainable
            for module in [self.layer_prefixes, self.aggregator]:
                for param in module.parameters():
                    param.requires_grad = True
            
            if hasattr(self, 'complexity_estimator'):
                for param in self.complexity_estimator.parameters():
                    param.requires_grad = True
            
            self.global_input_prefixes.requires_grad = True
            
            print("Stage 2 training: Base model frozen, ParScale components active")
        else:
            # Unfreeze all parameters for stage 1
            for param in self.parameters():
                param.requires_grad = True
            
            print("Stage 1 training: All parameters active")
    
    def forward_stream(
        self, 
        x: torch.Tensor, 
        stream_idx: int, 
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for a single stream with prefix transformations.
        
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
            stream_idx: Index of the parallel stream
            attention_mask: Optional attention mask
            freqs_cis: Rotary position embeddings
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Add global input prefix for this stream
        prefix = self.global_input_prefixes[stream_idx].unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([prefix, x], dim=1)
        
        # Update attention mask if needed
        if attention_mask is not None:
            prefix_len = self.config.prefix_length
            # Extend mask for prefix tokens
            prefix_mask = torch.zeros(batch_size, 1, seq_len + prefix_len, prefix_len, device=x.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)
        
        # Process through transformer layers with stream-specific prefixes
        for layer_idx, layer in enumerate(self.base_model.transformer.h):
            # Get layer-specific prefixes for this stream
            layer_prefixes = self.layer_prefixes[f'layer_{layer_idx}'].get_prefixes(stream_idx)
            
            # Forward through layer (we'll need to modify the layer to accept prefixes)
            # For now, use standard forward
            x = layer(x, start_pos=0, freqs_cis=freqs_cis, mask=attention_mask)
        
        # Final layer norm
        x = self.base_model.transformer.ln_f(x)
        
        # Remove prefix from output
        x = x[:, self.config.prefix_length:, :]
        
        return x
    
    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None,
        use_all_streams: bool = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with parallel streams.
        
        Args:
            idx: Input token indices [batch_size, seq_len]
            targets: Target tokens for loss computation
            use_all_streams: Override dynamic inference
            
        Returns:
            logits: Output logits
            loss: Optional loss value
        """
        # In training stage 1, just use base model
        if self.training and self.training_stage == 1:
            return self.base_model(idx, targets)
        
        device = idx.device
        batch_size, seq_len = idx.shape
        
        # Determine number of active streams
        if use_all_streams is False:
            active_streams = 1
        elif use_all_streams is True:
            active_streams = self.P
        elif self.config.use_dynamic_inference and hasattr(self, 'complexity_estimator'):
            # Estimate complexity and determine active streams
            complexity = self.complexity_estimator(idx).mean()
            active_streams = max(1, int(self.P * complexity))
        else:
            active_streams = self.P
        
        # Get embeddings
        embeddings = self.base_model.transformer.wte(idx)
        embeddings = self.base_model.transformer.drop(embeddings)
        
        # Prepare rotary embeddings and mask
        with torch.no_grad():
            mask = None
            if seq_len > 1:
                mask = torch.full((seq_len, seq_len), float("-inf"), device=device).triu_(1)
            
            freqs_cis = self.base_model.freqs_cis[:seq_len].detach()
        
        # Process through parallel streams
        parallel_outputs = []
        
        for stream_idx in range(active_streams):
            # Clone embeddings for this stream
            stream_embeddings = embeddings.clone()
            
            # Forward through this stream
            stream_output = self.forward_stream(
                stream_embeddings, 
                stream_idx, 
                attention_mask=mask,
                freqs_cis=freqs_cis
            )
            
            parallel_outputs.append(stream_output)
        
        # Stack outputs: [batch_size, P_active, seq_len, d_model]
        parallel_outputs = torch.stack(parallel_outputs, dim=1)
        
        # Aggregate outputs
        if active_streams == 1:
            # Skip aggregation for single stream
            aggregated = parallel_outputs.squeeze(1)
        else:
            # Pad if using fewer streams
            if active_streams < self.P:
                padding = torch.zeros(
                    batch_size, self.P - active_streams, seq_len, self.config.n_embd,
                    device=device, dtype=parallel_outputs.dtype
                )
                parallel_outputs = torch.cat([parallel_outputs, padding], dim=1)
            
            aggregated = self.aggregator(parallel_outputs)
        
        # Compute logits
        logits = self.base_model.lm_head(aggregated)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Standard cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=-100
            )
            
            # Add diversity regularization in stage 2
            if self.training and self.training_stage == 2 and active_streams > 1:
                div_loss = diversity_loss(parallel_outputs[:, :active_streams])
                loss = loss + self.config.diversity_weight * div_loss
        
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Configure optimizers for ParScale training."""
        if self.training_stage == 1:
            # Stage 1: Use base model's optimizer configuration
            return self.base_model.configure_optimizers(
                weight_decay, learning_rate, betas, device_type
            )
        else:
            # Stage 2: Only optimize ParScale components
            parscale_params = []
            
            # Collect ParScale parameters
            for module in [self.layer_prefixes, self.aggregator]:
                parscale_params.extend(module.parameters())
            
            parscale_params.append(self.global_input_prefixes)
            
            if hasattr(self, 'complexity_estimator'):
                parscale_params.extend(self.complexity_estimator.parameters())
            
            # Create optimizer
            optimizer = torch.optim.AdamW(
                parscale_params,
                lr=learning_rate,
                betas=betas,
                weight_decay=weight_decay
            )
            
            return optimizer
    
    @torch.no_grad()
    def generate(
        self, 
        idx: torch.Tensor, 
        max_new_tokens: int, 
        temperature: float = 1.0, 
        top_k: Optional[int] = None,
        use_all_streams: bool = True
    ) -> torch.Tensor:
        """Generate text using ParScale inference."""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Get logits from forward pass
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self.forward(idx_cond, use_all_streams=use_all_streams)
            
            # Extract last position logits
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def analyze_stream_diversity(self) -> Dict[str, float]:
        """Analyze diversity of parallel streams."""
        with torch.no_grad():
            # Analyze input prefix diversity
            input_prefixes = self.global_input_prefixes
            input_sim = torch.mm(
                F.normalize(input_prefixes.view(self.P, -1), dim=1),
                F.normalize(input_prefixes.view(self.P, -1), dim=1).t()
            )
            
            # Mask diagonal
            mask = 1 - torch.eye(self.P, device=input_sim.device)
            avg_input_similarity = (input_sim * mask).sum() / mask.sum()
            
            # Analyze latent prefix diversity for each layer
            layer_similarities = []
            for layer_idx in range(self.config.n_layer):
                prefixes = self.layer_prefixes[f'layer_{layer_idx}'].get_all_prefixes()
                
                # Average similarity across Q, K, V
                for prefix_type in ['q', 'k', 'v']:
                    p = prefixes[prefix_type]
                    p_norm = F.normalize(p.view(self.P, -1), dim=1)
                    sim = torch.mm(p_norm, p_norm.t())
                    layer_similarities.append((sim * mask).sum() / mask.sum())
            
            avg_layer_similarity = torch.stack(layer_similarities).mean()
            
            return {
                'input_similarity': avg_input_similarity.item(),
                'layer_similarity': avg_layer_similarity.item(),
                'effective_streams': self.P * (1 - avg_layer_similarity.item())
            }
    
    def dynamic_inference(
        self, 
        idx: torch.Tensor, 
        complexity_threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Perform dynamic inference based on input complexity.
        
        Args:
            idx: Input token indices
            complexity_threshold: Override default complexity threshold
            
        Returns:
            logits: Output logits
            active_streams: Number of streams used
        """
        if complexity_threshold is None:
            complexity_threshold = self.config.complexity_threshold
        
        # Estimate complexity
        with torch.no_grad():
            complexity = self.complexity_estimator(idx).mean().item()
        
        # Determine active streams
        if complexity < complexity_threshold:
            active_streams = max(1, int(self.P * complexity))
        else:
            active_streams = self.P
        
        # Forward with determined streams
        logits, _ = self.forward(idx, use_all_streams=(active_streams == self.P))
        
        return logits, active_streams
    
    def efficient_batch_inference(
        self, 
        batch_idx: torch.Tensor,
        group_by_length: bool = True
    ) -> torch.Tensor:
        """
        Efficient batched inference with length-based grouping.
        
        Args:
            batch_idx: Batch of input sequences [batch_size, seq_len]
            group_by_length: Whether to group by sequence length
            
        Returns:
            logits: Output logits maintaining original order
        """
        batch_size = batch_idx.shape[0]
        
        if not group_by_length:
            # Standard forward pass
            logits, _ = self.forward(batch_idx)
            return logits
        
        # Calculate actual lengths (excluding padding)
        lengths = (batch_idx != 0).sum(dim=1)
        
        # Sort by length
        sorted_lengths, sorted_indices = torch.sort(lengths)
        sorted_idx = batch_idx[sorted_indices]
        
        # Group sequences by similar lengths
        outputs = []
        current_group = []
        current_length = sorted_lengths[0].item()
        
        for i in range(batch_size):
            seq_length = sorted_lengths[i].item()
            
            # Start new group if length difference > 10%
            if seq_length > current_length * 1.1 or i == batch_size - 1:
                if current_group:
                    # Process current group
                    group_tensor = torch.stack(current_group)
                    group_logits, _ = self.forward(group_tensor)
                    outputs.append(group_logits)
                
                # Start new group
                current_group = [sorted_idx[i]]
                current_length = seq_length
            else:
                current_group.append(sorted_idx[i])
        
        # Process last group if not empty
        if current_group and len(outputs) < batch_size:
            group_tensor = torch.stack(current_group)
            group_logits, _ = self.forward(group_tensor)
            outputs.append(group_logits)
        
        # Concatenate all outputs
        all_logits = torch.cat(outputs, dim=0)
        
        # Restore original order
        unsorted_indices = torch.argsort(sorted_indices)
        return all_logits[unsorted_indices]
    
    def forward_with_cache(
        self,
        idx: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass with KV caching for efficient generation.
        
        This is a placeholder - full implementation would require modifying
        the MLA attention mechanism to support caching.
        """
        # For now, use standard forward without caching
        logits, loss = self.forward(idx)
        return logits, None


def create_parscale_mla(
    base_model: Optional[MLAModel] = None,
    size: str = 'small',
    parallel_streams: int = 8,
    **kwargs
) -> ParScaleMLA:
    """
    Create a ParScale-MLA model.
    
    Args:
        base_model: Pre-trained base MLA model (for stage 2 training)
        size: Model size if creating from scratch
        parallel_streams: Number of parallel streams (P)
        **kwargs: Additional configuration parameters
        
    Returns:
        ParScaleMLA model
    """
    if base_model is None:
        # Create new model from scratch
        from models.models.mla_model import create_mla_model
        base_model = create_mla_model(size=size, **kwargs)
    
    # Create ParScale config
    config = ParScaleMLAConfig(
        parallel_streams=parallel_streams,
        **kwargs
    )
    
    # Create ParScale model
    model = ParScaleMLA(base_model=base_model, config=config)
    
    return model