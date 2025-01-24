import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import custom_fwd, custom_bwd
import torch.utils.checkpoint as checkpoint
from model import CausalSelfAttention
from typing import Optional
import inspect
import gc

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
        
        # Router projection with memory-efficient initialization
        self.router = nn.Linear(input_dim, num_experts, bias=False)
        with torch.no_grad():
            nn.init.normal_(self.router.weight, mean=0.0, std=0.001)

        # Pre-register buffers with fixed size and pin memory for faster transfer
        self.register_buffer('_indices_buffer', torch.empty(0), persistent=False)
        self.register_buffer('_zeros_buffer', torch.empty(0), persistent=False)
        self.register_buffer('_dispatch_buffer', torch.empty(0), persistent=False)
        self.register_buffer('_ones_buffer', torch.empty(0), persistent=False)

    def _ensure_buffer_size(self, batch_size, seq_len):
        """Ensure buffers are correctly sized"""
        combined_size = batch_size * seq_len
        device = self.router.weight.device

        if self._indices_buffer.numel() < combined_size * self.k:
            self._indices_buffer = torch.arange(combined_size, device=device).unsqueeze(1).expand(-1, self.k)
        if self._zeros_buffer.numel() < self.num_experts:
            self._zeros_buffer = torch.zeros((self.num_experts,), device=device)
        if self._dispatch_buffer.numel() < combined_size * self.num_experts:
            self._dispatch_buffer = torch.zeros((combined_size, self.num_experts), device=device)
        if self._ones_buffer.numel() < combined_size:
            self._ones_buffer = torch.ones((combined_size,), dtype=torch.float32, device=device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        combined_batch_size = batch_size * seq_len
        device = x.device
        
        # Ensure buffers are correctly sized
        self._ensure_buffer_size(batch_size, seq_len)
        
        # Get buffer slices
        indices = self._indices_buffer[:combined_batch_size]
        dispatch_mask = self._dispatch_buffer[:combined_batch_size]
        expert_counts = self._zeros_buffer.clone()
        ones = self._ones_buffer[:combined_batch_size]
        
        # Clear dispatch mask
        dispatch_mask.zero_()
        
        # Calculate expert capacity
        capacity = int(self.capacity_factor * combined_batch_size * self.k / self.num_experts)
        
        # Compute router logits and probabilities using memory-efficient operations
        x_reshaped = x.view(-1, x.size(-1))
        with torch.cuda.amp.autocast(enabled=True):
            router_logits = self.router(x_reshaped)
            routing_weights = F.softmax(router_logits, dim=-1)
        
        # Get top-k experts and weights
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.k, dim=-1)
        
        # Normalize weights
        top_k_weights_sum = top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights / (top_k_weights_sum + 1e-8)
        
        # Create routing tensor using optimized scatter operations
        for k_idx in range(self.k):
            expert_idx = top_k_indices[:, k_idx]
            mask = expert_counts[expert_idx] < capacity
            expert_counts.scatter_add_(0, expert_idx[mask], ones[mask])
            dispatch_mask[indices[mask, k_idx], expert_idx[mask]] = top_k_weights[mask, k_idx]
        
        # Handle unrouted tokens efficiently
        unrouted = dispatch_mask.sum(dim=-1) == 0
        if torch.any(unrouted):
            least_loaded = (capacity - expert_counts).argmax()
            dispatch_mask[unrouted, least_loaded] = 1.0
        
        # Normalize dispatch mask
        dispatch_mask_sum = dispatch_mask.sum(dim=-1, keepdim=True)
        normalized_dispatch_mask = dispatch_mask / (dispatch_mask_sum + 1e-8)
        
        # Compute auxiliary losses efficiently
        router_z_loss = torch.mean(torch.square(router_logits))
        expert_counts = normalized_dispatch_mask.sum(0)
        target_count = float(combined_batch_size * self.k / self.num_experts)
        counts_scaled = expert_counts / float(combined_batch_size)
        target_scaled = target_count / float(combined_batch_size)
        load_balance_loss = torch.mean(torch.square(counts_scaled - target_scaled))
        
        # Combine losses with detached tensors
        router_loss = (0.001 * router_z_loss.detach() + 
                      0.001 * load_balance_loss.detach())
        
        # Clean up any remaining references
        del router_logits, routing_weights, top_k_weights, top_k_indices
        torch.cuda.empty_cache()
        
        return routing_weights.detach(), normalized_dispatch_mask, router_loss

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
        with torch.amp.autocast(enabled=False, device_type='cuda'):
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
        
        # Shared MLP backbone
        self.shared_mlp = SharedExpertMLP(config)
        
        # Expert-specific adaptation parameters
        # Utiliser les mêmes dimensions que SharedExpertMLP
        self.hidden_dim = 2 * config.n_embd  # Match SharedExpertMLP
        self.adapt_dim = self.hidden_dim // 16  # Match SharedExpertMLP
        
        # Expert adapters with matching dimensions
        self.expert_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.adapt_dim, self.adapt_dim, bias=False),
                nn.LayerNorm(self.adapt_dim)
            ) for _ in range(num_experts)
        ])
        
        # Projections with matching dimensions
        self.expert_proj = nn.Linear(self.adapt_dim, self.hidden_dim, bias=False)
        self.output_proj = nn.Linear(self.hidden_dim, config.n_embd, bias=False)
        
        # Initialize with small values
        adapt_scale = 0.01 / (self.adapt_dim ** 0.5)
        for adapter in self.expert_adapters:
            nn.init.normal_(adapter[0].weight, mean=0.0, std=adapt_scale)
        nn.init.normal_(self.expert_proj.weight, mean=0.0, std=adapt_scale)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=adapt_scale)

    def forward(self, x: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        
        # Get shared representations
        shared_output = self.shared_mlp(x)  # [B, S, n_embd]
        
        # Reshape expert_weights to match batch and sequence dimensions
        expert_weights = expert_weights.view(batch_size, seq_len, -1)  # [B, S, num_experts]
        
        # Apply expert-specific adaptations
        expert_outputs = []
        for i, adapter in enumerate(self.expert_adapters):
            if expert_weights[:, :, i].any():
                mask = expert_weights[:, :, i] > 0
                if mask.any():
                    # Process only tokens routed to this expert
                    expert_x = x[mask]
                    # Use shared_mlp's pre_adapt to ensure dimension matching
                    expert_hidden = self.shared_mlp.pre_adapt(expert_x)  # [masked_tokens, adapt_dim]
                    expert_hidden = adapter(expert_hidden)  # [masked_tokens, adapt_dim]
                    
                    # Project with matching dimensions
                    expert_hidden = self.expert_proj(expert_hidden)  # [masked_tokens, hidden_dim]
                    expert_hidden = self.output_proj(expert_hidden)  # [masked_tokens, n_embd]
                    
                    # Scale output by expert weights
                    curr_output = shared_output.clone()
                    mask_expanded = mask.unsqueeze(-1)  # [B, S, 1]
                    
                    # Add expert contribution
                    expert_contribution = torch.zeros_like(curr_output)  # [B, S, n_embd]
                    expert_contribution[mask] = expert_hidden  # Now dimensions match
                    
                    # Scale contribution and combine
                    curr_output = torch.where(
                        mask_expanded,
                        curr_output + 0.1 * expert_contribution,  # Reduce expert influence
                        curr_output
                    )
                    weight_expanded = expert_weights[:, :, i].unsqueeze(-1)  # [B, S, 1]
                    expert_outputs.append(curr_output * weight_expanded)
        
        # Combine outputs
        if expert_outputs:
            return sum(expert_outputs)
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
    def __init__(self, config, num_experts: int = 8, k: int = 2, capacity_factor: float = 1.25, hierarchical: bool = False):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.k = k
        
        # Router (hierarchical or standard)
        if hierarchical:
            self.router = HierarchicalRouter(config.n_embd, num_experts, k=self.k, group_size=4)
        else:
            self.router = Router(config.n_embd, num_experts, k=self.k, capacity_factor=capacity_factor)
        
        # Replace individual experts with ExpertGroup
        self.expert_group = ExpertGroup(config, num_experts)
        
        self.norm = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        batch_size, seq_len, hidden_dim = x.shape
        
        # Normalize input
        normalized = self.norm(x)
        
        # Get routing weights and dispatch mask
        combine_weights, dispatch_mask, router_loss = self.router(normalized)
        
        # Process through expert group
        output = self.expert_group(normalized, dispatch_mask)
        
        return output, router_loss

class MoEBlock(nn.Module):
    """
    Transformer block that uses MoE instead of standard MLP.
    """
    def __init__(self, config, num_experts: int = 8, k: int = 2):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.moe = MoELayer(config, num_experts=num_experts, k=k)
        self.use_checkpoint = True

    def forward(self, x: torch.Tensor, key_value: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        # Attention
        attn_output = self.attn(self.ln_1(x), key_value=key_value)
        x = x + attn_output
        
        # MoE
        if self.use_checkpoint and self.training:
            moe_output, router_loss = checkpoint.checkpoint(
                self.moe,
                self.ln_2(x),
                use_reentrant=False
            )
        else:
            moe_output, router_loss = self.moe(self.ln_2(x))
        
        x = x + moe_output
        
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

    def forward(self, encoder_idx: torch.Tensor, decoder_idx: torch.Tensor, 
                targets: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass of the MoE encoder-decoder model.
        
        Args:
            encoder_idx: Input tokens for encoder [batch_size, encoder_seq_len]
            decoder_idx: Input tokens for decoder [batch_size, decoder_seq_len]
            targets: Optional target tokens [batch_size, decoder_seq_len]
            
        Returns:
            logits: Output logits
            loss: Optional loss value if targets provided
            router_loss: Combined routing loss from all MoE layers
        """
        device = encoder_idx.device
        
        # Ensure input has correct shape
        if encoder_idx.dim() == 1:
            encoder_idx = encoder_idx.unsqueeze(0)
        if decoder_idx.dim() == 1:
            decoder_idx = decoder_idx.unsqueeze(0)
        
        # Process encoder input
        encoder_pos = torch.arange(0, encoder_idx.size(1), device=device)
        encoder_emb = self.shared_embedding(encoder_idx)
        encoder_pos_emb = self.shared_pos_embedding(encoder_pos)
        
        # Adjust dimensions for broadcasting
        encoder_pos_emb = encoder_pos_emb.unsqueeze(0).expand(encoder_idx.size(0), -1, -1)
        
        x = self.encoder.drop(encoder_emb + encoder_pos_emb)
        
        total_router_loss = torch.tensor(0.0, device=device)
        
        # Encoder forward pass
        for block in self.encoder.h:
            x, router_loss = block(x)
            total_router_loss = total_router_loss + router_loss
            
        encoder_output = self.encoder.ln_f(x)
        
        # Process decoder input
        decoder_pos = torch.arange(0, decoder_idx.size(1), device=device)
        decoder_emb = self.shared_embedding(decoder_idx)
        decoder_pos_emb = self.shared_pos_embedding(decoder_pos)
        
        # Adjust dimensions for broadcasting
        decoder_pos_emb = decoder_pos_emb.unsqueeze(0).expand(decoder_idx.size(0), -1, -1)
        
        x = self.decoder.drop(decoder_emb + decoder_pos_emb)
        
        # Decoder forward pass with cross-attention
        for i, (block, cross_attn) in enumerate(zip(self.decoder.h, self.cross_attention)):
            # Self-attention
            attn_output = block.attn(block.ln_1(x))
            x = x + attn_output
            
            # Cross-attention
            cross_x = self.cross_ln[i](x)
            cross_output = cross_attn(cross_x, key_value=encoder_output)
            x = x + cross_output
            
            # MoE layer
            moe_input = block.ln_2(x)
            moe_out, router_loss = block.moe(moe_input)
            x = x + moe_out
            total_router_loss = total_router_loss + router_loss
            
        x = self.decoder.ln_f(x)
        
        # Calculate logits and loss
        if targets is not None:
            logits = self.lm_head(x)
            
            # Scale down the router loss by the number of MoE layers
            num_moe_layers = len(self.encoder.h) + len(self.decoder.h)
            avg_router_loss = total_router_loss / num_moe_layers
            
            # Calculate cross entropy loss using reshape instead of view
            logits_2d = logits.reshape(-1, logits.size(-1))
            targets_1d = targets.reshape(-1)
            
            ce_loss = F.cross_entropy(
                logits_2d,
                targets_1d,
                ignore_index=-1
            )
            
            # Combine losses with a very small coefficient for router loss
            loss = ce_loss + 0.0001 * avg_router_loss
        else:
            # For generation, only compute logits for the last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss, total_router_loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, 
                temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """
        Generate text using the MoE model.
        """
        # Ensure input has correct shape
        
        print(idx)
        
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        
        # Encode input sequence once
        encoder_pos = torch.arange(0, idx.size(1), device=idx.device)
        encoder_emb = self.shared_embedding(idx)
        encoder_pos_emb = self.shared_pos_embedding(encoder_pos)
        
        # Adjust dimensions for broadcasting
        encoder_pos_emb = encoder_pos_emb.unsqueeze(0).expand(idx.size(0), -1, -1)
        
        x = self.encoder.drop(encoder_emb + encoder_pos_emb)
        
        # Encoder forward pass
        for block in self.encoder.h:
            x, _ = block(x)
            
        encoder_output = self.encoder.ln_f(x)
        
        # Create prompt-aware initial state for decoder
        prompt_summary = torch.mean(encoder_output, dim=1, keepdim=True)  # [B, 1, C]
        
        # Initialize decoder input with BOS token + prompt
        bos_token = torch.full(
            (idx.size(0), 1),
            self.shared_embedding.weight.size(0)-2,  # BOS token
            dtype=torch.long,
            device=idx.device
        )
        # Feed the entire prompt to decoder 
        decoder_idx = torch.cat([bos_token, idx], dim=1)
        
        # Track generated tokens for repetition detection
        last_tokens = []
        repetition_penalty = 1.5  # Increased from 1.2
        min_tokens_before_repetition = 5  # Increased from 3
        
        # Generate tokens autoregressively
        for _ in range(max_new_tokens):
            # Ensure we don't exceed maximum sequence length
            if decoder_idx.size(1) > self.decoder_config.block_size:
                decoder_idx = decoder_idx[:, -self.decoder_config.block_size:]
            
            # Decoder forward pass
            decoder_pos = torch.arange(0, decoder_idx.size(1), device=decoder_idx.device)
            decoder_emb = self.shared_embedding(decoder_idx)
            decoder_pos_emb = self.shared_pos_embedding(decoder_pos)
            
            # Add prompt context to decoder embeddings
            decoder_emb = decoder_emb + 0.3 * prompt_summary.expand(-1, decoder_emb.size(1), -1)
            
            # Adjust dimensions for broadcasting
            decoder_pos_emb = decoder_pos_emb.unsqueeze(0).expand(decoder_idx.size(0), -1, -1)
            
            # Process decoder input
            decoder_x = self.decoder.drop(decoder_emb + decoder_pos_emb)
            
            # Decoder layers with cross-attention
            for i, (block, cross_attn) in enumerate(zip(self.decoder.h, self.cross_attention)):
                # Self-attention
                attn_output = block.attn(block.ln_1(decoder_x))
                decoder_x = decoder_x + attn_output
                
                # Cross-attention with encoder output
                cross_x = self.cross_ln[i](decoder_x)
                # Explicitly use encoder output in cross-attention
                cross_output = cross_attn(
                    cross_x, 
                    key_value=encoder_output,
                    is_generation=True
                )
                decoder_x = decoder_x + cross_output
                
                # MoE layer
                moe_input = block.ln_2(decoder_x)
                moe_out, _ = block.moe(moe_input)
                decoder_x = decoder_x + moe_out
            
            decoder_x = self.decoder.ln_f(decoder_x)
            
            # Get logits for next token
            logits = self.lm_head(decoder_x[:, -1:])
            logits = logits.reshape(decoder_x.size(0), -1)  # Use reshape instead of squeeze
            
            # Apply repetition penalty
            if len(last_tokens) >= min_tokens_before_repetition:
                for token in set(last_tokens[-min_tokens_before_repetition:]):
                    logits[:, token] = logits[:, token] / repetition_penalty
            
            # Apply temperature and top-k sampling
            logits = logits / (temperature + 1e-5)  # Add small epsilon for stability
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Add length penalty to discourage very short sequences
            if decoder_idx.size(1) < 10:  # Minimum desired length
                length_penalty = torch.ones_like(logits) * -0.5
                logits = logits + length_penalty
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Update last tokens
            last_tokens.append(next_token.item())
            if len(last_tokens) > min_tokens_before_repetition * 2:
                last_tokens.pop(0)
            
            # Stop if we generate an EOS token
            if (next_token == self.shared_embedding.weight.size(0)-1).any():
                break
            
            # Append to sequence
            decoder_idx = torch.cat((decoder_idx, next_token), dim=1)
        
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
        # Encoder flops
        encoder_flops = batch_size * self.encoder_config.block_size * self.encoder_config.n_layer * (
            # Attention
            8 * self.encoder_config.n_embd * self.encoder_config.block_size +
            # MLP (assuming average expert utilization of k/num_experts)
            2 * 4 * self.encoder_config.n_embd * self.encoder_config.n_embd * (2/8)
        )
        
        # Decoder flops
        decoder_flops = batch_size * self.decoder_config.block_size * self.decoder_config.n_layer * (
            # Self-attention
            8 * self.decoder_config.n_embd * self.decoder_config.block_size +
            # Cross-attention
            8 * self.decoder_config.n_embd * self.encoder_config.block_size +
            # MLP (assuming average expert utilization of k/num_experts)
            2 * 4 * self.decoder_config.n_embd * self.decoder_config.n_embd * (2/8)
        )
        
        total_flops = encoder_flops + decoder_flops
        
        # Get device compute capability
        if hasattr(torch.cuda, 'get_device_capability'):
            device = next(self.parameters()).device
            if device.type == 'cuda':
                gpu_flops = {
                    (7, 0): 14e12,    # A100
                    (8, 0): 19e12,    # A100
                    (8, 6): 36e12,    # H100
                    (9, 0): 39e12     # H200
                }.get(torch.cuda.get_device_capability(device), 14e12)  # Default to A100
            else:
                gpu_flops = 14e12  # Default to A100 if not on GPU
        else:
            gpu_flops = 14e12  # Default if can't detect
        
        mfu = total_flops / (dt * gpu_flops)
        return mfu 

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure the AdamW optimizer with weight decay.
        Handles parameter groups for experts and non-expert parameters separately.
        """
        # Collect all parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Create parameter groups
        expert_params = []
        router_params = []
        other_params = []
        
        for name, param in param_dict.items():
            if any(expert_type in name.lower() for expert_type in ['expert_group', 'shared_mlp', 'expert_proj', 'expert_adapters']):
                expert_params.append(param)
            elif 'router' in name.lower():
                router_params.append(param)
            else:
                other_params.append(param)
        
        # Debug print
        print("\nParameter classification:")
        for name, param in param_dict.items():
            if any(expert_type in name.lower() for expert_type in ['expert_group', 'shared_mlp', 'expert_proj', 'expert_adapters']):
                print(f"Expert param: {name} - {param.numel():,} parameters")
            elif 'router' in name.lower():
                print(f"Router param: {name} - {param.numel():,} parameters")
            else:
                print(f"Other param: {name} - {param.numel():,} parameters")
        
        # Create optimizer groups with different hyperparameters
        optim_groups = [
            # Expert parameters
            {
                'params': expert_params,
                'weight_decay': weight_decay,
                'lr': learning_rate
            },
            # Router parameters (lower learning rate)
            {
                'params': router_params,
                'weight_decay': 0.0,  # No weight decay for router
                'lr': learning_rate * 0.1  # Lower learning rate for stability
            },
            # Other parameters
            {
                'params': other_params,
                'weight_decay': weight_decay,
                'lr': learning_rate
            }
        ]
        
        # Print parameter counts
        num_expert_params = sum(p.numel() for p in expert_params)
        num_router_params = sum(p.numel() for p in router_params)
        num_other_params = sum(p.numel() for p in other_params)
        
        print(f"\nParameter counts:")
        print(f"Expert parameters: {num_expert_params:,}")
        print(f"Router parameters: {num_router_params:,}")
        print(f"Other parameters: {num_other_params:,}")
        
        # Create AdamW optimizer with fused implementation if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra_args
        )
        
        print(f"Using fused AdamW: {use_fused}")
        
        return optimizer 