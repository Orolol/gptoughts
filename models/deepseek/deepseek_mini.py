import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import custom_fwd, custom_bwd
import torch.utils.checkpoint as checkpoint
from typing import Optional, Tuple, Dict, Any, Union, List
import inspect
import gc
import time
from collections import defaultdict
from contextlib import nullcontext

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self._norm(x.float()).type_as(x)

class DeepSeekMiniConfig:
    """Configuration class for DeepSeek Mini model"""
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        head_dim: int = 128,
        intermediate_size: int = 4096,
        num_experts: int = 32,
        num_experts_per_token: int = 4,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        layernorm_epsilon: float = 1e-6,
        kv_compression_dim: int = 128,
        query_compression_dim: int = 384,
        rope_head_dim: int = 32,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        bias: bool = False,
        expert_bias_init: float = 0.0,
        expert_bias_update_speed: float = 0.001,
        expert_balance_factor: float = 0.0001,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = True,
        use_return_dict: bool = True,
        router_z_loss_coef: float = 0.001,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.layernorm_epsilon = layernorm_epsilon
        self.kv_compression_dim = kv_compression_dim
        self.query_compression_dim = query_compression_dim
        self.rope_head_dim = rope_head_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.bias = bias
        self.expert_bias_init = expert_bias_init
        self.expert_bias_update_speed = expert_bias_update_speed
        self.expert_balance_factor = expert_balance_factor
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_cache = use_cache
        self.use_return_dict = use_return_dict
        self.router_z_loss_coef = router_z_loss_coef

class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings with decoupled head dimension"""
    
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: int = 10000, rope_head_dim: int = 32):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.rope_head_dim = rope_head_dim
        
        # Generate and cache cos/sin values
        inv_freq = 1.0 / (base ** (torch.arange(0, rope_head_dim, 2).float() / rope_head_dim))
        t = torch.arange(max_position_embeddings).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache cos and sin values [1, 1, seq_len, dim]
        self.register_buffer('cos_cached', emb.cos().view(1, 1, max_position_embeddings, rope_head_dim), persistent=False)
        self.register_buffer('sin_cached', emb.sin().view(1, 1, max_position_embeddings, rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[-2]
            
        # Get cos and sin values for current sequence length
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        return cos, sin

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    # Ensure all tensors have compatible shapes
    # q, k: [batch, heads, seq_len, head_dim]
    # cos, sin: [1, 1, seq_len, head_dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

class MLACompression(nn.Module):
    """Multi-Head Latent Attention Compression module"""
    
    def __init__(self, config: DeepSeekMiniConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        
        # Compression projections - ajuster les dimensions pour correspondre à head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=config.bias)
        
        # Layer norms
        self.q_norm = RMSNorm(config.head_dim)
        self.k_norm = RMSNorm(config.head_dim)
        self.v_norm = RMSNorm(config.head_dim)
        
        # Scaling factors
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project with correct dimensions
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape pour multi-head attention
        q = q.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        
        # Apply layer norm per head
        q = self.q_norm(q) * self.scale
        k = self.k_norm(k) * self.scale
        v = self.v_norm(v) * self.scale
        
        # Apply dropout
        q = self.dropout(q)
        k = self.dropout(k)
        v = self.dropout(v)
        
        return q, k, v

class MLAAttention(nn.Module):
    """Multi-Head Latent Attention with efficient inference through low-rank joint compression"""
    
    def __init__(self, config: DeepSeekMiniConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.dropout = config.attention_dropout
        
        # MLA compression
        self.compression = MLACompression(config)
        
        # Output projection
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        
        # Rotary embeddings for decoupled head dimension
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            config.head_dim
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Special scaled initialization to improve stability
        std = 0.02 / math.sqrt(2.0 * self.config.num_hidden_layers)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=std)
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Get compressed latent representations
        q_latent, k_latent, v_latent = self.compression(hidden_states)
        
        # Reshape for multi-head attention
        q_latent = q_latent.view(batch_size, seq_length, self.num_attention_heads, -1).transpose(1, 2)
        k_latent = k_latent.view(batch_size, seq_length, self.num_attention_heads, -1).transpose(1, 2)
        v_latent = v_latent.view(batch_size, seq_length, self.num_attention_heads, -1).transpose(1, 2)
        
        # Apply rotary embeddings to query and key latents
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        cos, sin = self.rotary_emb(position_ids, seq_length)
        q_latent, k_latent = apply_rotary_pos_emb(q_latent, k_latent, cos, sin)
        
        # Handle past key values
        if past_key_value is not None:
            k_latent = torch.cat([past_key_value[0], k_latent], dim=2)
            v_latent = torch.cat([past_key_value[1], v_latent], dim=2)
        
        past_key_value = (k_latent, v_latent) if use_cache else None
        
        # Compute attention scores
        attn_weights = torch.matmul(q_latent, k_latent.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v_latent)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        outputs = (attn_output, past_key_value)
        if output_attentions:
            outputs += (attn_weights,)
            
        return outputs

class SharedExpert(nn.Module):
    """Shared expert module used in DeepSeek Mini"""
    
    def __init__(self, config: DeepSeekMiniConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self._init_weights()

    def _init_weights(self):
        for module in [self.w1, self.w2, self.w3]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w1(x)) * self.w3(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x

class ExpertRouter(nn.Module):
    """Router module with auxiliary-loss-free load balancing"""
    
    def __init__(self, config: DeepSeekMiniConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        
        # Router projection
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
        # Expert bias terms for load balancing
        self.expert_bias = nn.Parameter(torch.ones(config.num_experts) * config.expert_bias_init)
        
        # Load balancing parameters
        self.bias_update_speed = config.expert_bias_update_speed
        self.balance_factor = config.expert_balance_factor
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_length, hidden_dim = hidden_states.shape
        
        # Compute router logits
        router_logits = self.router(hidden_states)
        
        # Add expert bias terms
        router_logits = router_logits + self.expert_bias.unsqueeze(0).unsqueeze(0)
        
        # Get top-k experts
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(
            router_probs, 
            self.num_experts_per_token,
            dim=-1
        )
        
        # Normalize probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Create dispatch tensors
        dispatch_mask = torch.zeros_like(router_probs)
        dispatch_mask.scatter_(-1, top_k_indices, top_k_probs)
        
        # Compute load balancing loss
        # Ideal load would be uniform distribution across experts
        expert_load = router_probs.mean(dim=[0, 1])
        target_load = torch.ones_like(expert_load) / self.num_experts
        balance_loss = F.kl_div(
            expert_load.log(),
            target_load,
            reduction='batchmean',
            log_target=False
        )
        
        # Update expert bias terms during training
        if self.training:
            with torch.no_grad():
                load_diff = expert_load - target_load
                self.expert_bias.data -= self.bias_update_speed * load_diff
        
        return dispatch_mask, balance_loss * self.balance_factor, top_k_indices

class DeepSeekMiniMoE(nn.Module):
    """MoE layer with shared and routed experts"""
    
    def __init__(self, config: DeepSeekMiniConfig):
        super().__init__()
        self.config = config
        
        # Shared expert
        self.shared_expert = SharedExpert(config)
        
        # Routed experts
        self.experts = nn.ModuleList([
            SharedExpert(config) for _ in range(config.num_experts)
        ])
        
        # Router
        self.router = ExpertRouter(config)
        
        # Layer norm
        self.norm = RMSNorm(config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        
        # Get routing probabilities and expert assignments
        dispatch_mask, balance_loss, expert_indices = self.router(hidden_states)
        
        # Process with shared expert
        shared_output = self.shared_expert(hidden_states)
        
        # Process with routed experts
        batch_size, seq_length, hidden_dim = hidden_states.shape
        expert_output = torch.zeros_like(hidden_states)
        
        # Compute expert outputs efficiently
        flat_hidden = hidden_states.view(-1, hidden_dim)
        for i, expert in enumerate(self.experts):
            # Get tokens assigned to this expert
            expert_mask = dispatch_mask[..., i].view(-1)
            if not expert_mask.any():
                continue
                
            # Process tokens with expert
            expert_input = flat_hidden[expert_mask.bool()]
            expert_output_i = expert(expert_input)
            
            # Combine outputs
            expert_output.view(-1, hidden_dim)[expert_mask.bool()] += expert_output_i * expert_mask[expert_mask.bool()].unsqueeze(-1)
        
        # Combine shared and routed expert outputs
        output = shared_output + expert_output
        output = self.dropout(output)
        
        return output + residual, balance_loss 

class DeepSeekMiniBlock(nn.Module):
    """Transformer block with MLA attention and MoE FFN"""
    
    def __init__(self, config: DeepSeekMiniConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Pre-attention norm
        self.norm1 = RMSNorm(config.hidden_size)
        
        # MLA attention
        self.attn = MLAAttention(config)
        
        # MoE or standard FFN based on layer index
        if layer_idx < 3:  # First three layers use standard FFN
            self.mlp = SharedExpert(config)
            self.is_moe_layer = False
        else:
            self.mlp = DeepSeekMiniMoE(config)
            self.is_moe_layer = True
        
        # Dropouts
        self.dropout1 = nn.Dropout(config.hidden_dropout)
        self.dropout2 = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        # Attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        
        hidden_states = residual + self.dropout1(attn_output)
        
        # MoE or FFN
        if self.is_moe_layer:
            layer_output, balance_loss = self.mlp(hidden_states)
        else:
            layer_output = self.mlp(hidden_states)
            balance_loss = torch.tensor(0.0, device=hidden_states.device)
        
        hidden_states = hidden_states + self.dropout2(layer_output)
        
        if use_cache:
            outputs = (hidden_states,) + outputs + (balance_loss,)
        else:
            outputs = (hidden_states, balance_loss) + outputs[1:]
            
        return outputs

class DeepSeekMini(nn.Module):
    """DeepSeek Mini model with MLA attention and MoE"""
    
    def __init__(self, config: DeepSeekMiniConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            DeepSeekMiniBlock(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm_f = RMSNorm(config.hidden_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to residual projections
        self.apply(self._init_residual_weights)
        
        # Initialize timing stats
        self.timing_stats = None
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_residual_weights(self, module):
        if isinstance(module, (MLAAttention, SharedExpert)):
            for name, param in module.named_parameters():
                if "o_proj" in name or "w2" in name:
                    nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2.0 * self.config.num_hidden_layers))

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def set_gradient_checkpointing(self, value: bool):
        """Enable or disable gradient checkpointing"""
        self.gradient_checkpointing = value
        # Also set gradient checkpointing for all transformer blocks
        for layer in self.layers:
            if hasattr(layer, 'set_gradient_checkpointing'):
                layer.set_gradient_checkpointing(value)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict[str, torch.Tensor]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # Create attention mask if needed
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=hidden_states.device)
        
        # Convert attention mask to attention bias
        attention_bias = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        attention_bias = attention_bias.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_length]

        # Initialize lists to collect outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_balance_losses = []
        next_cache = () if use_cache else None

        # Forward through layers
        for idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_bias,
                    position_ids,
                    None,  # past_key_value
                    output_attentions,
                    False,  # use_cache
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_bias,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            balance_loss = layer_outputs[1] if len(layer_outputs) > 1 else torch.tensor(0.0, device=hidden_states.device)
            all_balance_losses.append(balance_loss)

            if use_cache:
                next_cache = next_cache + (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[3 if use_cache else 2],)

        # Final layer norm
        hidden_states = self.norm_f(hidden_states)

        # Add hidden states from the last layer if needed
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Calculate total balance loss
        # Ensure all elements in all_balance_losses are tensors, not tuples
        tensor_balance_losses = []
        for loss in all_balance_losses:
            if isinstance(loss, tuple):
                # Si c'est un tuple, on extrait le tenseur de perte (généralement le premier élément)
                tensor_balance_losses.append(loss[0] if isinstance(loss[0], torch.Tensor) else torch.tensor(0.0, device=hidden_states.device))
            else:
                # Si c'est déjà un tenseur, on l'ajoute directement
                tensor_balance_losses.append(loss)
        
        total_balance_loss = torch.stack(tensor_balance_losses).mean()

        if not return_dict:
            return tuple(v for v in [
                hidden_states,
                next_cache,
                all_hidden_states,
                all_self_attns,
                total_balance_loss
            ] if v is not None)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
            "balance_loss": total_balance_loss
        }

    def _gradient_checkpointing_func(self, func, *args, **kwargs):
        """Helper function for gradient checkpointing"""
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        return checkpoint.checkpoint(create_custom_forward(func), *args, **kwargs)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.LongTensor:
        batch_size, cur_length = input_ids.shape
        device = input_ids.device
        
        # Vérifier qu'au moins un des paramètres max_length ou max_new_tokens est fourni
        if max_length is None and max_new_tokens is None:
            raise ValueError("Vous devez spécifier soit max_length soit max_new_tokens")
            
        # Calculer max_length à partir de max_new_tokens si nécessaire
        if max_length is None:
            max_length = cur_length + max_new_tokens
            
        if pad_token_id is None:
            pad_token_id = 0  # Default to 0
        if eos_token_id is None:
            eos_token_id = 2  # Default to 2
            
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        
        past_key_values = None
        cur_length = input_ids.shape[1]
        
        while cur_length < max_length:
            # Prepare model inputs
            model_inputs = {
                "input_ids": input_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
            }
            
            # Forward pass
            outputs = self.forward(**model_inputs)
            next_token_logits = outputs["last_hidden_state"][:, -1, :]
            past_key_values = outputs["past_key_values"]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(input_ids[i].tolist()):
                        next_token_logits[i, previous_token] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Update unfinished sequences
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).squeeze())
            
            # Stop if all sequences are finished
            if unfinished_sequences.max() == 0:
                break
                
            # Update input_ids
            input_ids = torch.cat([input_ids, next_tokens], dim=1)
            cur_length = input_ids.shape[1]
        
        return input_ids 

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
        
        # Create optimizer groups with different hyperparameters
        optim_groups = [
            # Expert parameters
            {
                'params': expert_params,
                'weight_decay': weight_decay,
                'lr': learning_rate
            },
            # Router parameters (no weight decay)
            {
                'params': router_params,
                'weight_decay': 0.0,
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

    def set_timing_stats(self, timing_stats):
        """Set the timing stats object for detailed profiling"""
        self.timing_stats = timing_stats
        # Also set timing stats for all transformer blocks
        for layer in self.layers:
            if hasattr(layer, 'set_timing_stats'):
                layer.set_timing_stats(timing_stats)

    def estimate_mfu(self, batch_size: int, dt: float) -> float:
        """Estimate model flops utilization (MFU) in percentage."""
        # Calculate total flops per token
        total_flops = batch_size * self.config.max_position_embeddings * self.config.num_hidden_layers * (
            # Attention flops
            8 * self.config.hidden_size * self.config.max_position_embeddings +
            # MLP flops (assuming average expert utilization of num_experts_per_token/num_experts)
            2 * 4 * self.config.hidden_size * self.config.intermediate_size * 
            (self.config.num_experts_per_token / self.config.num_experts)
        )
        
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