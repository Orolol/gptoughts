import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import custom_fwd, custom_bwd
import torch.utils.checkpoint as checkpoint
from models.blocks.attention import CausalSelfAttention
from typing import Optional
import inspect
import gc
import time
from collections import defaultdict
from contextlib import nullcontext
import threading
from queue import Queue

# Import des fonctions de calcul de FLOPS et MFU
from train.train_utils import estimate_mfu as utils_estimate_mfu

class Router(nn.Module):
    """
    Router module that determines which expert should process each token.
    Uses top-k routing with load balancing.
    """
    def __init__(self, input_dim: int, num_experts: int, k: int = 2, capacity_factor: float = 1.25):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor
        
        # Simpler router architecture with better initialization
        self.router = nn.Sequential(
            nn.Linear(input_dim, num_experts, bias=False),
            nn.LayerNorm(num_experts)  # Normalize router logits
        )
        
        # Better initialization for routing
        nn.init.normal_(self.router[0].weight, mean=0.0, std=0.01)
        
        # Increased temperature for softer routing decisions
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        # Adjusted loss coefficients
        self.router_z_loss_coef = 0.01  # Increased from 0.001
        self.load_balance_coef = 0.01   # Increased from 0.001

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        combined_batch_size = batch_size * seq_len
        
        # Compute router logits
        router_logits = self.router(x.view(-1, self.input_dim))
        
        # Scale logits by learned temperature
        router_logits = router_logits / (self.temperature.abs() + 1e-6)
        
        # Compute routing probabilities with stable softmax
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Get top-k routing weights and indices
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.k, dim=-1)
        
        # Normalize top-k weights
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Create dispatch mask.
        dispatch_mask = torch.zeros_like(routing_weights)
        dispatch_mask.scatter_(
            dim=-1,
            index=top_k_indices,
            src=top_k_weights
        )
        
        # Compute load balancing loss
        # Ideal load would be uniform distribution across experts
        ideal_load = torch.ones_like(routing_weights.mean(0)) / self.num_experts
        actual_load = routing_weights.mean(0)
        load_balance_loss = F.kl_div(
            actual_load.log(),
            ideal_load,
            reduction='batchmean',
            log_target=False
        )
        
        # Router z-loss to encourage exploration
        router_z_loss = torch.square(router_logits).mean()
        
        # Combine losses with adjusted coefficients
        router_loss = (
            self.router_z_loss_coef * router_z_loss +
            self.load_balance_coef * load_balance_loss
        )
        
        # Reshape dispatch mask for batch processing
        dispatch_mask = dispatch_mask.view(batch_size, seq_len, -1)
        
        return routing_weights.detach(), dispatch_mask, router_loss

class SharedExpertMLP(nn.Module):
    """
    Expert MLP with partial weight sharing.
    Each expert shares a common backbone but has unique adaptation layers.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Dimensions avec facteur plus petit pour plus de stabilité
        self.hidden_dim = 2 * config.n_embd  # Réduit de 4x à 2x
        
        # Shared parameters
        self.shared_up = nn.Linear(config.n_embd, self.hidden_dim, bias=config.bias)
        self.shared_gate = nn.Linear(config.n_embd, self.hidden_dim, bias=config.bias)
        self.shared_down = nn.Linear(self.hidden_dim, config.n_embd, bias=config.bias)
        
        # Smaller adaptation dimensions
        self.adapt_dim = self.hidden_dim // 16  # Encore plus petit pour la stabilité
        self.pre_adapt = nn.Linear(config.n_embd, self.adapt_dim, bias=config.bias)
        self.post_adapt = nn.Linear(self.hidden_dim, self.adapt_dim, bias=config.bias)
        
        # Layer norm pour stabiliser
        self.adapt_norm = nn.LayerNorm(self.adapt_dim)
        
        # Projection avec plus petit facteur d'échelle
        self.adapt_proj = nn.Linear(self.adapt_dim, self.hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        
        # Main computation with shared weights
        up = self.shared_up(x)
        gate = self.shared_gate(x)
        hidden = F.silu(gate) * up
        
        # Adaptation pathway avec plus de normalisation
        adapt_in = self.adapt_norm(self.pre_adapt(x))
        adapt_out = self.adapt_norm(self.post_adapt(hidden))
        
        # Compute adaptation weights avec clipping
        adapt_weights = torch.bmm(adapt_in, adapt_out.transpose(1, 2))
        adapt_weights = torch.clamp(adapt_weights, -5.0, 5.0)  # Prevent exploding values
        adapt_weights = F.silu(adapt_weights)
        
        # Apply adaptation avec gradient scaling
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            adapt = torch.bmm(adapt_weights.float(), adapt_in.float())
        adapt = self.adapt_proj(adapt.to(x.dtype))
        
        # Scale down adaptation contribution
        adapt = 0.1 * adapt  # Reduce influence of adaptation
        
        # Combine main path with adaptation
        hidden = hidden + adapt
        
        return self.dropout(self.shared_down(hidden))

    def _init_weights(self):
        # Smaller initialization for more stability
        scale = 1 / (self.config.n_embd ** 0.5)
        for layer in [self.shared_up, self.shared_gate, self.shared_down]:
            nn.init.normal_(layer.weight, mean=0.0, std=scale)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        
        # Even smaller scale for adaptation layers
        adapt_scale = scale * 0.01
        for layer in [self.pre_adapt, self.post_adapt, self.adapt_proj]:
            nn.init.normal_(layer.weight, mean=0.0, std=adapt_scale)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.zeros_(layer.bias)

class ExpertGroup(nn.Module):
    """
    A group of experts that share some weights but maintain unique characteristics
    through adaptation layers.
    """
    def __init__(self, config, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.config = config
        
        # Define dimensions
        self.hidden_dim = 2 * config.n_embd
        self.adapt_dim = self.hidden_dim // 16
        
        # Shared MLP backbone with larger capacity
        self.shared_mlp = SharedExpertMLP(config)
        
        # Expert-specific adaptation with parallel computation
        self.expert_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.adapt_dim, self.adapt_dim, bias=False),
                nn.LayerNorm(self.adapt_dim)
            ) for _ in range(num_experts)
        ])
        
        # Projections for dimension matching
        self.expert_proj = nn.Linear(self.adapt_dim, self.hidden_dim, bias=False)
        self.output_proj = nn.Linear(self.hidden_dim, config.n_embd, bias=False)
        
        # Enable parallel computation
        self.parallel_adapters = True
        
        # Initialize with small values
        adapt_scale = 0.01 / (self.adapt_dim ** 0.5)
        for adapter in self.expert_adapters:
            nn.init.normal_(adapter[0].weight, mean=0.0, std=adapt_scale)
        nn.init.normal_(self.expert_proj.weight, mean=0.0, std=adapt_scale)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=adapt_scale)

    def forward(self, x: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        
        # Get shared representations
        shared_output = self.shared_mlp(x)
        
        if self.parallel_adapters:
            # Process all experts in parallel
            expert_outputs = []
            masks = []
            
            # Ensure expert_weights has correct shape [batch_size, seq_len, num_experts]
            if expert_weights.dim() == 2:
                expert_weights = expert_weights.view(batch_size, seq_len, -1)
            
            # Create masks for each expert
            for i in range(self.num_experts):
                mask = expert_weights[:, :, i] > 0
                if mask.any():
                    masks.append(mask)
                    expert_x = x[mask]
                    # Process through adapter and projections
                    expert_hidden = self.shared_mlp.pre_adapt(expert_x)
                    expert_hidden = self.expert_adapters[i](expert_hidden)
                    expert_hidden = self.expert_proj(expert_hidden)
                    expert_hidden = self.output_proj(expert_hidden)
                    expert_outputs.append(expert_hidden)
            
            # Combine expert outputs efficiently
            if expert_outputs:
                combined_output = torch.zeros_like(shared_output)
                for mask, expert_output in zip(masks, expert_outputs):
                    combined_output[mask] = expert_output
                
                return shared_output + 0.1 * combined_output
        
        return shared_output

class HierarchicalRouter(nn.Module):
    """
    Hierarchical router with two gating steps: 
    Step 1: route tokens to a group of experts
    Step 2: pick a specific expert within that group
    """
    def __init__(self, input_dim: int, num_experts: int, k: int = 2, group_size: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.k = k
        self.group_size = group_size
        self.num_groups = max(1, num_experts // group_size)

        # Gate 1 routes tokens to a group
        self.router_group = nn.Linear(input_dim, self.num_groups, bias=False)
        # Gate 2 routes tokens to an expert within each group
        self.router_expert = nn.Linear(input_dim, group_size, bias=False)

        # Initialize
        nn.init.normal_(self.router_group.weight, mean=0.0, std=0.001)
        nn.init.normal_(self.router_expert.weight, mean=0.0, std=0.001)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [batch_size, seq_len, hidden_dim]
        b, seq_len, _ = x.shape
        x_flat = x.view(-1, x.size(-1))  # [b * seq_len, hidden_dim]

        # Step 1: route to group
        group_logits = self.router_group(x_flat)                # [b*seq_len, num_groups]
        group_probs = F.softmax(group_logits, dim=-1)           # [b*seq_len, num_groups]
        top_k_group_weights, top_k_group_indices = torch.topk(group_probs, k=1, dim=-1)
        # We keep only the best group (k=1), but you can keep more.
        chosen_group = top_k_group_indices.squeeze(dim=-1)      # [b*seq_len]
        chosen_group_weight = top_k_group_weights.squeeze(dim=-1)

        # Step 2: route within the chosen group
        # Build a local index to pass to second gate
        local_expert_logits = self.router_expert(x_flat)        # [b*seq_len, group_size]
        local_expert_probs = F.softmax(local_expert_logits, dim=-1)

        # Flatten group + local index => unique expert index
        # group i occupies expert indices [i*group_size ... (i+1)*group_size-1]
        expert_offset = chosen_group * self.group_size
        # pick top-k within group
        top_k_local_w, top_k_local_idx = torch.topk(local_expert_probs, self.k, dim=-1)
        # combine the group offset
        top_k_expert_idx = expert_offset.unsqueeze(1) + top_k_local_idx  # [b*seq_len, k]

        # Merge weighting
        top_k_local_w_sum = top_k_local_w.sum(dim=-1, keepdim=True)
        top_k_local_w = top_k_local_w / (top_k_local_w_sum + 1e-7)
        # final weight includes group weight
        final_weights = (chosen_group_weight.unsqueeze(1) * top_k_local_w)  # [b*seq_len, k]

        # Build dispatch mask [b*seq_len, num_experts]
        dispatch_mask = x.new_zeros(x_flat.size(0), self.num_experts)
        for i in range(self.k):
            dispatch_mask.scatter_(1, top_k_expert_idx[:, i:i+1], final_weights[:, i:i+1])

        # Basic load-balancing: measure how many tokens each expert gets
        expert_load = dispatch_mask.sum(dim=0)  # [num_experts]
        # Simple MSE-based load-balancing loss
        target_load = expert_load.sum() / self.num_experts
        load_balance_loss = F.mse_loss(expert_load, torch.full_like(expert_load, target_load))

        # Z-loss for router logits
        z_loss = (group_logits**2).mean() + (local_expert_logits**2).mean()
        router_loss = 0.001 * (load_balance_loss + z_loss)

        return final_weights.detach(), dispatch_mask, router_loss

class MoELayer(nn.Module):
    """
    Mixture of Experts layer that combines multiple expert MLPs with a router.
    """
    def __init__(self, config, num_experts: int = 8, k: int = 2):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.k = k
        
        # Initialize router with adjusted parameters
        self.router = Router(
            input_dim=config.n_embd,
            num_experts=num_experts,
            k=k,
            capacity_factor=1.5  # Increased capacity factor
        )
        
        # Expert group with shared parameters
        self.expert_group = ExpertGroup(config, num_experts)
        
        # Add layer norm before routing
        self.norm = nn.LayerNorm(config.n_embd)
        
        # Add residual scaling
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        
        # Normalize input
        normalized = self.norm(x)
        
        # Get routing weights and dispatch mask
        routing_weights, dispatch_mask, router_loss = self.router(normalized)
        
        # Process through expert group
        expert_output = self.expert_group(normalized, dispatch_mask)
        
        # Add scaled residual connection
        output = residual + self.residual_scale * expert_output
        
        return output, router_loss

class MoEBlock(nn.Module):
    """
    Block combining attention and MoE layers
    """
    def __init__(self, config, num_experts, k):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        
        # MoE components
        self.router = Router(config.n_embd, num_experts, k)
        self.expert_group = ExpertGroup(config, num_experts)
        
        # Router loss coefficient
        self.router_z_loss = 0.001
        
        # Gradient checkpointing flag
        self.use_checkpoint = False

    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.ln_1(x))
        
        # MoE layer
        moe_input = self.ln_2(x)
        routing_weights, dispatch_mask, router_loss = self.router(moe_input)
        
        # Ensure dispatch_mask has correct shape
        if dispatch_mask.dim() == 2:
            batch_size, seq_len = x.shape[:2]
            dispatch_mask = dispatch_mask.view(batch_size, seq_len, -1)
            
        expert_output = self.expert_group(moe_input, dispatch_mask)
        x = x + expert_output
        
        return x, router_loss

class MoEEncoderDecoderGPT(nn.Module):
    """
    Encoder-Decoder model that uses MoE blocks instead of standard transformer blocks.
    """
    def __init__(self, encoder_config, decoder_config, num_experts: int = 8, k: int = 2):
        super().__init__()
        
        # Ensure compatible dimensions
        assert encoder_config.n_embd == decoder_config.n_embd
        assert encoder_config.vocab_size == decoder_config.vocab_size
        assert encoder_config.block_size == decoder_config.block_size
        
        # Shared embeddings
        self.shared_embedding = nn.Embedding(encoder_config.vocab_size, encoder_config.n_embd)
        self.shared_pos_embedding = nn.Embedding(encoder_config.block_size, encoder_config.n_embd)
        
        # Create encoder and decoder with MoE blocks
        self.encoder = nn.ModuleDict(dict(
            drop = nn.Dropout(encoder_config.dropout),
            h = nn.ModuleList([MoEBlock(encoder_config, num_experts, k) for _ in range(encoder_config.n_layer)]),
            ln_f = nn.LayerNorm(encoder_config.n_embd)
        ))
        
        self.decoder = nn.ModuleDict(dict(
            drop = nn.Dropout(decoder_config.dropout),
            h = nn.ModuleList([MoEBlock(decoder_config, num_experts, k) for _ in range(decoder_config.n_layer)]),
            ln_f = nn.LayerNorm(decoder_config.n_embd)
        ))
        
        # Output head (shared with input embeddings)
        self.lm_head = nn.Linear(decoder_config.n_embd, decoder_config.vocab_size, bias=False)
        self.lm_head.weight = self.shared_embedding.weight
        
        # Cross-attention
        self.cross_attention = nn.ModuleList([
            CausalSelfAttention(decoder_config) 
            for _ in range(decoder_config.n_layer)
        ])
        
        self.cross_ln = nn.ModuleList([
            nn.LayerNorm(decoder_config.n_embd)
            for _ in range(decoder_config.n_layer)
        ])
        
        # Save configs
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        
        # Initialize weights
        self.apply(self._init_weights)
        
        self.timing_stats = None  # Pour stocker l'objet timing_stats

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def set_timing_stats(self, timing_stats):
        """Set the timing stats object for detailed profiling"""
        self.timing_stats = timing_stats

    def forward(self, encoder_inputs: torch.Tensor, decoder_inputs: torch.Tensor, 
                targets: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        device = encoder_inputs.device
        total_router_loss = torch.tensor(0.0, device=device)
        
        track = self.timing_stats.track if self.timing_stats is not None else nullcontext
        
        with track("forward"):
            with track("encoder"):
                # Encoder
                with track("embeddings"):
                    encoder_pos = torch.arange(0, encoder_inputs.size(1), device=device)
                    encoder_emb = self.shared_embedding(encoder_inputs)
                    encoder_pos_emb = self.shared_pos_embedding(encoder_pos)
                    encoder_pos_emb = encoder_pos_emb.unsqueeze(0).expand(encoder_inputs.size(0), -1, -1)
                    x = self.encoder.drop(encoder_emb + encoder_pos_emb)
                
                # Process through layers
                for i, block in enumerate(self.encoder.h):
                    with track("attention"):
                        x = x + block.attn(block.ln_1(x))
                    
                    with track("moe"):
                        moe_input = block.ln_2(x)
                        with track("router"):
                            routing_weights, dispatch_mask, router_loss = block.router(moe_input)
                        with track("experts"):
                            expert_output = block.expert_group(moe_input, dispatch_mask)
                            x = x + expert_output
                        total_router_loss = total_router_loss + router_loss
                
                with track("final_norm"):
                    encoder_output = self.encoder.ln_f(x)
            
            with track("decoder"):
                # Embeddings
                with track("embeddings"):
                    decoder_pos = torch.arange(0, decoder_inputs.size(1), device=device)
                    decoder_emb = self.shared_embedding(decoder_inputs)
                    decoder_pos_emb = self.shared_pos_embedding(decoder_pos)
                    decoder_pos_emb = decoder_pos_emb.unsqueeze(0).expand(decoder_inputs.size(0), -1, -1)
                    x = self.decoder.drop(decoder_emb + decoder_pos_emb)
                
                # Process through layers
                for i, (block, cross_attn) in enumerate(zip(self.decoder.h, self.cross_attention)):
                    with track("self_attention"):
                        x = x + block.attn(block.ln_1(x))
                    
                    with track("cross_attention"):
                        cross_x = self.cross_ln[i](x)
                        cross_output = cross_attn(cross_x, key_value=encoder_output)
                        x = x + cross_output
                    
                    with track("moe"):
                        moe_input = block.ln_2(x)
                        with track("router"):
                            routing_weights, dispatch_mask, router_loss = block.router(moe_input)
                        with track("experts"):
                            expert_output = block.expert_group(moe_input, dispatch_mask)
                            x = x + expert_output
                        total_router_loss = total_router_loss + router_loss
                
                with track("final_norm"):
                    x = self.decoder.ln_f(x)
            
            with track("head"):
                if targets is not None:
                    logits = self.lm_head(x)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        ignore_index=-1
                    )
                else:
                    logits = self.lm_head(x[:, [-1], :])
                    loss = None
        
        return logits, loss, total_router_loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, 
                temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """
        Generate text using the MoE model.
        """
        # Ensure input has correct shape and is on the correct device
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        
        # Encode input sequence once
        encoder_pos = torch.arange(0, idx.size(1), device=idx.device)
        encoder_emb = self.shared_embedding(idx)
        encoder_pos_emb = self.shared_pos_embedding(encoder_pos)
        encoder_pos_emb = encoder_pos_emb.unsqueeze(0).expand(idx.size(0), -1, -1)
        
        x = self.encoder.drop(encoder_emb + encoder_pos_emb)
        
        # Encoder forward pass
        for block in self.encoder.h:
            # Self-attention
            x = x + block.attn(block.ln_1(x))
            
            # MoE layer
            moe_input = block.ln_2(x)
            routing_weights, dispatch_mask, _ = block.router(moe_input)
            expert_output = block.expert_group(moe_input, dispatch_mask)
            x = x + expert_output
        
        encoder_output = self.encoder.ln_f(x)
        
        # Initialize decoder input with BOS token
        bos_token = torch.full(
            (idx.size(0), 1),
            self.shared_embedding.weight.size(0)-2,  # BOS token
            dtype=torch.long,
            device=idx.device
        )
        decoder_idx = bos_token
        
        # Generate tokens autoregressively
        for _ in range(max_new_tokens):
            # Ensure we don't exceed maximum sequence length
            if decoder_idx.size(1) > self.decoder_config.block_size:
                decoder_idx = decoder_idx[:, -self.decoder_config.block_size:]
            
            # Decoder forward pass
            decoder_pos = torch.arange(0, decoder_idx.size(1), device=decoder_idx.device)
            decoder_emb = self.shared_embedding(decoder_idx)
            decoder_pos_emb = self.shared_pos_embedding(decoder_pos)
            decoder_pos_emb = decoder_pos_emb.unsqueeze(0).expand(decoder_idx.size(0), -1, -1)
            
            decoder_x = self.decoder.drop(decoder_emb + decoder_pos_emb)
            
            # Decoder layers
            for layer_idx, (block, cross_attn, cross_ln) in enumerate(zip(
                self.decoder.h, 
                self.cross_attention, 
                self.cross_ln
            )):
                # Self-attention
                decoder_x = decoder_x + block.attn(block.ln_1(decoder_x))
                
                # Cross-attention
                cross_x = cross_ln(decoder_x)
                cross_output = cross_attn(
                    cross_x, 
                    key_value=encoder_output,
                    is_generation=True
                )
                decoder_x = decoder_x + cross_output
                
                # MoE layer
                moe_input = block.ln_2(decoder_x)
                routing_weights, dispatch_mask, _ = block.router(moe_input)
                expert_output = block.expert_group(moe_input, dispatch_mask)
                decoder_x = decoder_x + expert_output
            
            decoder_x = self.decoder.ln_f(decoder_x)
            
            # Get logits for next token
            logits = self.lm_head(decoder_x[:, -1:])
            
            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence and continue
            decoder_idx = torch.cat((decoder_idx, idx_next), dim=1)
        
        return decoder_idx

    def set_gradient_checkpointing(self, value: bool):
        """Set gradient checkpointing for all transformer blocks."""
        # Set checkpointing for encoder blocks
        for block in self.encoder.h:
            block.use_checkpoint = value
            
        # Set checkpointing for decoder blocks
        for block in self.decoder.h:
            block.use_checkpoint = value

    def estimate_mfu(self, batch_size: int, dt: float) -> float:
        """Estimate model flops utilization (MFU) in percentage."""
        # Utiliser la fonction importée de train_utils.py
        # Pour un modèle encoder-decoder, nous utilisons la longueur de séquence maximale
        # entre l'encodeur et le décodeur
        seq_length = max(self.encoder_config.block_size, self.decoder_config.block_size)
        
        return utils_estimate_mfu(
            model=self,
            batch_size=batch_size,
            seq_length=seq_length,
            dt=dt,
            dtype=torch.float16  # Utiliser fp16 comme demandé
        )

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure optimizer for MoE models with specialized parameter groups.
        
        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            betas: Adam beta parameters
            device_type: Device type ('cuda' or 'cpu')
            
        Returns:
            Configured optimizer
        """
        from models.optimizers import configure_optimizer_for_moe
        return configure_optimizer_for_moe(
            model=self,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
            device_type=device_type
        ) 