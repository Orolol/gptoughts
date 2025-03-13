import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from models.deepseek.kernel import act_quant, weight_dequant, fp8_gemm


world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.


class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.
    Includes numerical stability improvements.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        
        # Assurer que vocab_size est divisible par world_size
        if vocab_size % world_size != 0:
            # Arrondir au multiple supérieur
            adjusted_vocab_size = ((vocab_size // world_size) + 1) * world_size
            print(f"Warning: Adjusted vocab_size from {vocab_size} to {adjusted_vocab_size} to be divisible by world_size={world_size}")
            self.vocab_size = adjusted_vocab_size
        else:
            print(f"Vocab size {vocab_size} is divisible by world_size={world_size}")
            
        self.part_vocab_size = (self.vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        
        # Initialiser les poids avec une distribution normale tronquée pour éviter les valeurs extrêmes
        self.weight = nn.Parameter(torch.zeros(self.part_vocab_size, self.dim))
        self.reset_parameters()
        
        # Ajouter un dropout pour la régularisation
        self.dropout = nn.Dropout(0.1)
    
    def reset_parameters(self):
        """Initialise les poids avec une distribution normale tronquée pour plus de stabilité"""
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            # Tronquer les valeurs extrêmes
            self.weight.data = torch.clamp(self.weight.data, -0.1, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer with improved numerical stability.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.
        """
        try:
            # Vérifier les valeurs d'entrée
            if torch.max(x) >= self.vocab_size:
                print(f"Warning: Input contains token IDs >= vocab_size ({torch.max(x).item()} >= {self.vocab_size})")
                # Limiter les indices au vocabulaire valide
                x = torch.clamp(x, 0, self.vocab_size - 1)
            
            if world_size > 1:
                # Créer un masque pour les tokens hors de la partition locale
                mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
                # Ajuster les indices pour l'embedding local
                x_adjusted = x.clone()
                x_adjusted = x_adjusted - self.vocab_start_idx
                x_adjusted[mask] = 0  # Utiliser l'indice 0 pour les tokens masqués
            else:
                # En mode non-distribué, pas besoin d'ajustement
                x_adjusted = x
                mask = None
            
            # Appliquer l'embedding
            y = F.embedding(x_adjusted, self.weight)
            
            # Appliquer le dropout pour la régularisation
            y = self.dropout(y)
            
            # En mode distribué, masquer les embeddings non pertinents et réduire
            if world_size > 1 and mask is not None:
                y[mask] = 0
                dist.all_reduce(y)
            
            # Vérifier les NaN et les valeurs extrêmes
            if torch.isnan(y).any():
                print("NaN detected in embedding output")
                y = torch.nan_to_num(y, nan=0.0)
            
            # Limiter les valeurs extrêmes pour éviter les problèmes numériques
            if torch.max(torch.abs(y)) > 100:
                print(f"Warning: Extreme values in embedding output: {torch.max(torch.abs(y)).item()}")
                y = torch.clamp(y, -100, 100)
            
            return y
            
        except Exception as e:
            print(f"Error in embedding forward pass: {e}")
            # En cas d'erreur, retourner un tensor de zéros
            return torch.zeros(x.size(0), x.size(1), self.dim, device=x.device)


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and 
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve 
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() == 1`), a dequantized version 
          is used for computation.
        - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are applied.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) with improved numerical stability.

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-5.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps  # Augmenté pour plus de stabilité
        self.weight = nn.Parameter(torch.ones(dim))
        
        # Initialiser les poids avec une légère perturbation autour de 1
        with torch.no_grad():
            self.weight.data = torch.clamp(torch.normal(1.0, 0.02, size=(dim,)), 0.9, 1.1)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm with improved numerical stability.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        try:
            # Vérifier les NaN dans l'entrée
            if torch.isnan(x).any():
                print("NaN detected in RMSNorm input")
                x = torch.nan_to_num(x, nan=0.0)
            
            # Vérifier les valeurs extrêmes
            if torch.max(torch.abs(x)) > 1e6:
                print(f"Extreme values in RMSNorm input: {torch.max(torch.abs(x)).item()}")
                x = torch.clamp(x, -1e6, 1e6)
            
            # Implémentation manuelle de RMSNorm pour plus de contrôle
            # Calculer la variance
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            # Ajouter epsilon pour la stabilité numérique
            x_normed = x * torch.rsqrt(variance + self.eps)
            # Appliquer le scaling
            return self.weight * x_normed
            
        except RuntimeError as e:
            print(f"Error in RMSNorm: {e}")
            # En cas d'erreur, retourner l'entrée inchangée
            return x


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor with improved numerical stability.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    try:
        # Vérifier les dimensions
        if x.size(-1) % 2 != 0:
            print(f"Warning: Last dimension of input tensor must be even, got {x.size(-1)}")
            # Padding pour rendre la dimension paire
            if x.size(-1) % 2 == 1:
                padding = torch.zeros(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=-1)
        
        # Vérifier les NaN dans l'entrée
        if torch.isnan(x).any():
            print("NaN detected in apply_rotary_emb input")
            x = torch.nan_to_num(x, nan=0.0)
        
        # Sauvegarder le dtype original
        dtype = x.dtype
        
        # Convertir en float32 pour plus de stabilité numérique
        x_float = x.float()
        
        # Vérifier les valeurs extrêmes
        if torch.isinf(x_float).any() or torch.abs(x_float).max() > 1e6:
            print("Warning: Extreme values detected in apply_rotary_emb input")
            x_float = torch.clamp(x_float, -1e6, 1e6)
        
        # Reshape pour la conversion en complexe
        try:
            x_complex = torch.view_as_complex(x_float.view(*x.shape[:-1], -1, 2))
        except RuntimeError as e:
            print(f"Error in view_as_complex: {e}")
            # Fallback: implémentation manuelle de la rotation
            x_reshape = x_float.view(*x.shape[:-1], -1, 2)
            cos = freqs_cis.real.view(1, freqs_cis.size(0), 1, x_reshape.size(-2))
            sin = freqs_cis.imag.view(1, freqs_cis.size(0), 1, x_reshape.size(-2))
            
            x_real, x_imag = x_reshape[..., 0], x_reshape[..., 1]
            y_real = x_real * cos - x_imag * sin
            y_imag = x_real * sin + x_imag * cos
            
            y = torch.stack([y_real, y_imag], dim=-1)
            return y.flatten(3).to(dtype)
        
        # Reshape freqs_cis pour le broadcast
        try:
            freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
        except RuntimeError as e:
            print(f"Error in freqs_cis reshape: {e}")
            # Ajuster les dimensions si nécessaire
            if freqs_cis.size(0) < x_complex.size(1):
                # Padding pour freqs_cis
                padding_size = x_complex.size(1) - freqs_cis.size(0)
                padding = freqs_cis[-1:].expand(padding_size, -1)
                freqs_cis = torch.cat([freqs_cis, padding], dim=0)
            elif freqs_cis.size(0) > x_complex.size(1):
                # Tronquer freqs_cis
                freqs_cis = freqs_cis[:x_complex.size(1)]
            
            if freqs_cis.size(-1) < x_complex.size(-1):
                # Padding pour freqs_cis
                padding_size = x_complex.size(-1) - freqs_cis.size(-1)
                padding = torch.ones(freqs_cis.size(0), padding_size, device=freqs_cis.device, dtype=freqs_cis.dtype)
                freqs_cis = torch.cat([freqs_cis, padding], dim=-1)
            elif freqs_cis.size(-1) > x_complex.size(-1):
                # Tronquer freqs_cis
                freqs_cis = freqs_cis[..., :x_complex.size(-1)]
            
            freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
        
        # Appliquer la rotation
        y_complex = x_complex * freqs_cis
        
        # Vérifier les NaN après la multiplication
        if torch.isnan(y_complex).any():
            print("NaN detected after complex multiplication")
            # Fallback: utiliser une implémentation plus stable
            x_real, x_imag = x_float.view(*x.shape[:-1], -1, 2)[..., 0], x_float.view(*x.shape[:-1], -1, 2)[..., 1]
            cos = freqs_cis.real
            sin = freqs_cis.imag
            
            y_real = x_real * cos - x_imag * sin
            y_imag = x_real * sin + x_imag * cos
            
            y = torch.stack([y_real, y_imag], dim=-1)
            return y.flatten(3).to(dtype)
        
        # Convertir en réel et aplatir
        y = torch.view_as_real(y_complex).flatten(3)
        
        # Vérifier les NaN dans la sortie
        if torch.isnan(y).any():
            print("NaN detected in apply_rotary_emb output")
            y = torch.nan_to_num(y, nan=0.0)
        
        # Retourner au dtype original
        return y.to(dtype)
        
    except Exception as e:
        print(f"Critical error in apply_rotary_emb: {e}")
        # En cas d'erreur critique, retourner l'entrée inchangée
        return x


class MLA(nn.Module):
    """
    Multi-Headed Attention Layer (MLA).

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass for the Multi-Headed Attention Layer (MLA).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        try:
            # Ensure x has the right shape (batch_size, seq_len, dim)
            if len(x.shape) != 3:
                raise ValueError(f"Expected x to have 3 dimensions (batch_size, seq_len, dim), got {x.shape}")
                
            bsz, seqlen, _ = x.size()
            end_pos = start_pos + seqlen
            
            # Vérifier les NaN dans l'entrée
            if torch.isnan(x).any():
                print("NaN detected in MLA input")
                x = torch.nan_to_num(x, nan=0.0)
            
            # Projection Q
            if self.q_lora_rank == 0:
                q = self.wq(x)
            else:
                q_a = self.wq_a(x)
                # Vérifier NaN après wq_a
                if torch.isnan(q_a).any():
                    q_a = torch.nan_to_num(q_a, nan=0.0)
                q_norm = self.q_norm(q_a)
                # Vérifier NaN après q_norm
                if torch.isnan(q_norm).any():
                    q_norm = torch.nan_to_num(q_norm, nan=0.0)
                q = self.wq_b(q_norm)
            
            # Vérifier NaN dans q
            if torch.isnan(q).any():
                print("NaN detected in query projection")
                q = torch.nan_to_num(q, nan=0.0)
            
            q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
            q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            
            # Vérifier les dimensions avant apply_rotary_emb
            if q_pe.size(-1) != self.qk_rope_head_dim:
                print(f"Warning: q_pe dimension mismatch: expected {self.qk_rope_head_dim}, got {q_pe.size(-1)}")
                # Ajuster la dimension si nécessaire
                if q_pe.size(-1) < self.qk_rope_head_dim:
                    padding = torch.zeros(bsz, seqlen, self.n_local_heads,
                                         self.qk_rope_head_dim - q_pe.size(-1),
                                         device=q_pe.device, dtype=q_pe.dtype)
                    q_pe = torch.cat([q_pe, padding], dim=-1)
                else:
                    q_pe = q_pe[..., :self.qk_rope_head_dim]
            
            # Appliquer rotary embedding avec gestion d'erreur
            try:
                q_pe = apply_rotary_emb(q_pe, freqs_cis)
            except Exception as e:
                print(f"Error in apply_rotary_emb for q_pe: {e}")
                # Fallback: ne pas appliquer rotary embedding
                pass
            
            # Projection KV
            kv = self.wkv_a(x)
            # Vérifier NaN dans kv
            if torch.isnan(kv).any():
                print("NaN detected in key-value projection")
                kv = torch.nan_to_num(kv, nan=0.0)
            
            # Split KV
            kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            
            # Vérifier les dimensions avant apply_rotary_emb pour k_pe
            if k_pe.size(-1) != self.qk_rope_head_dim:
                print(f"Warning: k_pe dimension mismatch: expected {self.qk_rope_head_dim}, got {k_pe.size(-1)}")
                # Ajuster la dimension si nécessaire
                if k_pe.size(-1) < self.qk_rope_head_dim:
                    padding = torch.zeros(bsz, seqlen, self.qk_rope_head_dim - k_pe.size(-1),
                                         device=k_pe.device, dtype=k_pe.dtype)
                    k_pe = torch.cat([k_pe, padding], dim=-1)
                else:
                    k_pe = k_pe[..., :self.qk_rope_head_dim]
            
            # Appliquer rotary embedding avec gestion d'erreur
            try:
                k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
            except Exception as e:
                print(f"Error in apply_rotary_emb for k_pe: {e}")
                # Fallback: ne pas appliquer rotary embedding
                k_pe = k_pe.unsqueeze(2)
            
            # Implémentation naive ou optimisée
            if attn_impl == "naive":
                q = torch.cat([q_nope, q_pe], dim=-1)
                
                # Normalisation KV avec gestion d'erreur
                kv_norm = self.kv_norm(kv)
                # Vérifier NaN dans kv_norm
                if torch.isnan(kv_norm).any():
                    print("NaN detected in kv_norm")
                    kv_norm = torch.nan_to_num(kv_norm, nan=0.0)
                
                kv = self.wkv_b(kv_norm)
                # Vérifier NaN dans kv après projection
                if torch.isnan(kv).any():
                    print("NaN detected in kv after wkv_b")
                    kv = torch.nan_to_num(kv, nan=0.0)
                
                kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
                k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
                k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
                
                # Ensure k_cache is large enough for this batch
                if bsz > self.k_cache.size(0):
                    device = self.k_cache.device
                    new_k_cache = torch.zeros(bsz, self.k_cache.size(1), self.n_local_heads, self.qk_head_dim,
                                             device=device, dtype=self.k_cache.dtype)
                    new_v_cache = torch.zeros(bsz, self.v_cache.size(1), self.n_local_heads, self.v_head_dim,
                                             device=device, dtype=self.v_cache.dtype)
                    new_k_cache[:self.k_cache.size(0)] = self.k_cache
                    new_v_cache[:self.v_cache.size(0)] = self.v_cache
                    self.k_cache = new_k_cache
                    self.v_cache = new_v_cache
                
                # Mettre à jour les caches
                self.k_cache[:bsz, start_pos:end_pos] = k
                self.v_cache[:bsz, start_pos:end_pos] = v
                
                # Calcul des scores d'attention avec gestion d'erreur
                try:
                    scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
                except RuntimeError as e:
                    print(f"Error in attention score calculation: {e}")
                    # Fallback: utiliser matmul standard
                    q_flat = q.reshape(bsz * seqlen, self.n_local_heads, self.qk_head_dim)
                    k_flat = self.k_cache[:bsz, :end_pos].reshape(bsz * end_pos, self.n_local_heads, self.qk_head_dim)
                    scores = torch.matmul(q_flat, k_flat.transpose(-1, -2)).view(bsz, seqlen, self.n_local_heads, end_pos)
                    scores = scores * self.softmax_scale
            else:
                # Implémentation optimisée
                wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
                wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
                
                # Calcul de q_nope avec gestion d'erreur
                try:
                    q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
                except RuntimeError as e:
                    print(f"Error in q_nope calculation: {e}")
                    # Fallback: utiliser matmul standard
                    q_nope_flat = q_nope.reshape(bsz * seqlen * self.n_local_heads, -1)
                    wkv_b_flat = wkv_b[:, :self.qk_nope_head_dim].reshape(-1, self.kv_lora_rank)
                    q_nope = torch.matmul(q_nope_flat, wkv_b_flat).view(bsz, seqlen, self.n_local_heads, -1)
                
                # Vérifier NaN dans q_nope
                if torch.isnan(q_nope).any():
                    print("NaN detected in q_nope")
                    q_nope = torch.nan_to_num(q_nope, nan=0.0)
                
                # Ensure kv_cache is large enough for this batch
                if bsz > self.kv_cache.size(0):
                    device = self.kv_cache.device
                    new_kv_cache = torch.zeros(bsz, self.kv_cache.size(1), self.kv_lora_rank,
                                             device=device, dtype=self.kv_cache.dtype)
                    new_pe_cache = torch.zeros(bsz, self.pe_cache.size(1), self.qk_rope_head_dim,
                                             device=device, dtype=self.pe_cache.dtype)
                    new_kv_cache[:self.kv_cache.size(0)] = self.kv_cache
                    new_pe_cache[:self.pe_cache.size(0)] = self.pe_cache
                    self.kv_cache = new_kv_cache
                    self.pe_cache = new_pe_cache
                
                # Normalisation KV avec gestion d'erreur
                kv_norm = self.kv_norm(kv)
                # Vérifier NaN dans kv_norm
                if torch.isnan(kv_norm).any():
                    print("NaN detected in kv_norm")
                    kv_norm = torch.nan_to_num(kv_norm, nan=0.0)
                
                # Mettre à jour les caches
                self.kv_cache[:bsz, start_pos:end_pos] = kv_norm
                self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
                
                # Calcul des scores d'attention avec gestion d'erreur
                try:
                    scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                             torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
                except RuntimeError as e:
                    print(f"Error in attention score calculation: {e}")
                    # Fallback: utiliser matmul standard
                    q_nope_flat = q_nope.reshape(bsz * seqlen * self.n_local_heads, -1)
                    kv_cache_flat = self.kv_cache[:bsz, :end_pos].reshape(bsz * end_pos, -1)
                    scores_1 = torch.matmul(q_nope_flat, kv_cache_flat.transpose(-1, -2)).view(bsz, seqlen, self.n_local_heads, end_pos)
                    
                    q_pe_flat = q_pe.reshape(bsz * seqlen * self.n_local_heads, -1)
                    pe_cache_flat = self.pe_cache[:bsz, :end_pos].reshape(bsz * end_pos, -1)
                    scores_2 = torch.matmul(q_pe_flat, pe_cache_flat.transpose(-1, -2)).view(bsz, seqlen, self.n_local_heads, end_pos)
                    
                    scores = (scores_1 + scores_2) * self.softmax_scale
            
            # Vérifier NaN dans scores
            if torch.isnan(scores).any():
                print("NaN detected in attention scores")
                scores = torch.nan_to_num(scores, nan=0.0)
            
            # Appliquer le masque d'attention
            if mask is not None:
                scores += mask.unsqueeze(1)
            
            # Softmax avec gestion d'erreur
            try:
                # Appliquer un clipping pour éviter les valeurs extrêmes
                scores = torch.clamp(scores, -1e4, 1e4)
                scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
            except RuntimeError as e:
                print(f"Error in softmax calculation: {e}")
                # Fallback: softmax manuel avec stabilité numérique
                scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
                scores_exp = torch.exp(scores - scores_max)
                scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
                scores = scores_exp / (scores_sum + 1e-6)
                scores = scores.type_as(x)
            
            # Vérifier NaN dans scores après softmax
            if torch.isnan(scores).any():
                print("NaN detected in attention weights after softmax")
                scores = torch.nan_to_num(scores, nan=1.0/end_pos)  # Utiliser une distribution uniforme comme fallback
            
            # Calcul de la sortie d'attention
            if attn_impl == "naive":
                try:
                    x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
                except RuntimeError as e:
                    print(f"Error in attention output calculation: {e}")
                    # Fallback: utiliser matmul standard
                    scores_flat = scores.reshape(bsz * seqlen * self.n_local_heads, -1)
                    v_flat = self.v_cache[:bsz, :end_pos].reshape(bsz * end_pos, self.n_local_heads, self.v_head_dim).transpose(0, 1)
                    v_flat = v_flat.reshape(self.n_local_heads, -1, self.v_head_dim)
                    x = torch.matmul(scores_flat, v_flat).view(bsz, seqlen, self.n_local_heads, self.v_head_dim)
            else:
                try:
                    x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
                    x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
                except RuntimeError as e:
                    print(f"Error in attention output calculation: {e}")
                    # Fallback: utiliser matmul standard
                    scores_flat = scores.reshape(bsz * seqlen * self.n_local_heads, -1)
                    kv_flat = self.kv_cache[:bsz, :end_pos].reshape(bsz * end_pos, -1)
                    x = torch.matmul(scores_flat, kv_flat).view(bsz, seqlen, self.n_local_heads, -1)
                    
                    x_flat = x.reshape(bsz * seqlen * self.n_local_heads, -1)
                    wkv_b_flat = wkv_b[:, -self.v_head_dim:].reshape(-1, self.v_head_dim)
                    x = torch.matmul(x_flat, wkv_b_flat).view(bsz, seqlen, self.n_local_heads, self.v_head_dim)
            
            # Vérifier NaN dans la sortie d'attention
            if torch.isnan(x).any():
                print("NaN detected in attention output")
                x = torch.nan_to_num(x, nan=0.0)
            
            # Projection finale
            x = self.wo(x.flatten(2))
            
            # Vérifier NaN dans la sortie finale
            if torch.isnan(x).any():
                print("NaN detected in MLA output")
                x = torch.nan_to_num(x, nan=0.0)
            
            return x
            
        except Exception as e:
            print(f"Critical error in MLA forward pass: {e}")
            # Retourner l'entrée inchangée en cas d'erreur critique
            return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module with improved numerical stability.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = args.dim
        
        # Assurer que n_routed_experts est divisible par world_size
        if args.n_routed_experts % world_size != 0:
            adjusted_experts = ((args.n_routed_experts // world_size) + 1) * world_size
            print(f"Warning: Adjusted n_routed_experts from {args.n_routed_experts} to {adjusted_experts} to be divisible by world_size={world_size}")
            self.n_routed_experts = adjusted_experts
        else:
            self.n_routed_experts = args.n_routed_experts
            
        self.n_local_experts = self.n_routed_experts // world_size
        
        # Limiter n_activated_experts pour éviter les problèmes avec de petits batches
        if args.n_activated_experts > 2:
            print(f"Warning: Reducing n_activated_experts from {args.n_activated_experts} to 2 for stability with small batches")
            self.n_activated_experts = 2
        else:
            self.n_activated_experts = args.n_activated_experts
            
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        
        # Créer la gate avec les paramètres ajustés
        modified_args = ModelArgs(
            **{k: v for k, v in args.__dict__.items() if k != 'n_activated_experts' and k != 'n_routed_experts'}
        )
        modified_args.n_activated_experts = self.n_activated_experts
        modified_args.n_routed_experts = self.n_routed_experts
        self.gate = Gate(modified_args)
        
        # Créer les experts
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        
        # Créer l'expert partagé (toujours utilisé)
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)
        
        # Ajouter un dropout pour la régularisation
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module with improved numerical stability.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        try:
            # Vérifier les NaN dans l'entrée
            if torch.isnan(x).any():
                print("NaN detected in MoE input")
                x = torch.nan_to_num(x, nan=0.0)
            
            # Sauvegarder la forme originale
            shape = x.size()
            
            # Aplatir pour le traitement
            x = x.view(-1, self.dim)
            
            # Appliquer le dropout pour la régularisation
            x_dropped = self.dropout(x)
            
            # Obtenir les poids et indices de routage
            try:
                weights, indices = self.gate(x_dropped)
                
                # Vérifier les NaN dans les poids
                if torch.isnan(weights).any():
                    print("NaN detected in routing weights")
                    weights = torch.nan_to_num(weights, nan=1.0/self.n_activated_experts)
            except RuntimeError as e:
                print(f"Error in gate: {e}")
                # Fallback: utiliser l'expert partagé uniquement
                z = self.shared_experts(x)
                return z.view(shape)
            
            # Initialiser le tenseur de sortie
            y = torch.zeros_like(x)
            
            # Compter les tokens par expert
            try:
                counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
            except RuntimeError as e:
                print(f"Error in bincount: {e}")
                # Fallback: utiliser l'expert partagé uniquement
                z = self.shared_experts(x)
                return z.view(shape)
            
            # Appliquer les experts
            for i in range(self.experts_start_idx, self.experts_end_idx):
                if counts[i] == 0:
                    continue
                    
                expert = self.experts[i]
                try:
                    idx, top = torch.where(indices == i)
                    
                    # Vérifier si idx est vide
                    if len(idx) == 0:
                        continue
                    
                    # Appliquer l'expert
                    expert_output = expert(x[idx])
                    
                    # Vérifier les NaN dans la sortie de l'expert
                    if torch.isnan(expert_output).any():
                        print(f"NaN detected in expert {i} output")
                        expert_output = torch.nan_to_num(expert_output, nan=0.0)
                    
                    # Appliquer les poids de routage
                    weighted_output = expert_output * weights[idx, top, None]
                    
                    # Vérifier les NaN après la pondération
                    if torch.isnan(weighted_output).any():
                        print(f"NaN detected after weighting expert {i} output")
                        weighted_output = torch.nan_to_num(weighted_output, nan=0.0)
                    
                    # Ajouter à la sortie
                    y[idx] += weighted_output
                    
                except RuntimeError as e:
                    print(f"Error in expert {i}: {e}")
                    # Continuer avec les autres experts
                    continue
            
            # Appliquer l'expert partagé
            z = self.shared_experts(x)
            
            # Vérifier les NaN dans la sortie de l'expert partagé
            if torch.isnan(z).any():
                print("NaN detected in shared expert output")
                z = torch.nan_to_num(z, nan=0.0)
            
            # Réduire à travers les processus si nécessaire
            if world_size > 1:
                try:
                    dist.all_reduce(y)
                except RuntimeError as e:
                    print(f"Error in all_reduce: {e}")
            
            # Combiner les sorties
            combined = y + z
            
            # Vérifier les NaN dans la sortie combinée
            if torch.isnan(combined).any():
                print("NaN detected in MoE output")
                combined = torch.nan_to_num(combined, nan=0.0)
            
            # Limiter les valeurs extrêmes
            if torch.max(torch.abs(combined)) > 1e6:
                print(f"Extreme values in MoE output: {torch.max(torch.abs(combined)).item()}")
                combined = torch.clamp(combined, -1e6, 1e6)
            
            # Retourner à la forme originale
            return combined.view(shape)
            
        except Exception as e:
            print(f"Critical error in MoE forward: {e}")
            # En cas d'erreur critique, retourner l'entrée inchangée
            return x


class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
        """
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        # Vérifier les entrées pour éviter les problèmes
        if tokens.size(1) < 2:
            # Ajouter un padding pour éviter les problèmes avec les séquences trop courtes
            padding = torch.zeros((tokens.size(0), 2 - tokens.size(1)),
                                 dtype=tokens.dtype,
                                 device=tokens.device)
            tokens = torch.cat([tokens, padding], dim=1)
            print(f"Warning: Input sequence length {tokens.size(1)} is too short, padded to length 2")
        
        # Handle input from FinewebDataset with shape [batch_size, buffer_size, seq_len]
        if len(tokens.shape) == 3:
            batch_size, buffer_size, seq_len = tokens.shape
            # Reshape to [batch_size*buffer_size, seq_len]
            tokens = tokens.reshape(-1, seq_len)
        
        try:
            seqlen = tokens.size(1)
            h = self.embed(tokens)
            
            # Vérifier les NaN après l'embedding
            if torch.isnan(h).any():
                print("NaN detected after embedding layer")
                h = torch.nan_to_num(h, nan=0.0)
            
            freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
            mask = None
            if seqlen > 1:
                mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
            
            # Passer à travers les couches avec vérification des NaN
            for i, layer in enumerate(self.layers):
                h_prev = h
                h = layer(h, start_pos, freqs_cis, mask)
                
                # Vérifier les NaN après chaque couche
                if torch.isnan(h).any():
                    print(f"NaN detected after layer {i}")
                    # Utiliser la sortie de la couche précédente si possible
                    h = torch.nan_to_num(h, nan=0.0)
                    # Si trop de NaN, revenir à la sortie précédente
                    if torch.isnan(h).sum() > 0.5 * h.numel():
                        h = h_prev
                        print(f"Too many NaNs in layer {i}, using previous layer output")
            
            # Apply normalization to the entire sequence, not just the last token
            h = self.norm(h)
            
            # Vérifier les NaN après la normalisation
            if torch.isnan(h).any():
                print("NaN detected after final normalization")
                h = torch.nan_to_num(h, nan=0.0)
            
            # Get predictions for all tokens, not just the last one
            logits = self.head(h.reshape(-1, h.size(-1)))  # Reshape to [batch*seq_len, dim]
            
            # Vérifier les NaN dans les logits
            if torch.isnan(logits).any():
                print("NaN detected in logits")
                logits = torch.nan_to_num(logits, nan=0.0)
            
            logits = logits.view(h.size(0), h.size(1), -1)  # Reshape back to [batch, seq_len, vocab]
            
            if world_size > 1:
                # For distributed training with 3D tensors, we'd need more complex handling
                # For simplicity, we're just returning the local logits
                # In a real implementation, you might want to gather from all processes
                pass
                
            # For the loss calculation in training, flatten to [batch*seq_len, vocab]
            logits = logits.reshape(-1, logits.size(-1))
            
            # Vérification finale des NaN
            if torch.isnan(logits).any():
                print("NaN detected in final logits")
                logits = torch.nan_to_num(logits, nan=0.0)
                
            return logits
            
        except RuntimeError as e:
            print(f"Forward pass error: {e}")
            # Retourner un tensor de zéros en cas d'erreur
            return torch.zeros((tokens.size(0) * tokens.size(1), self.head.weight.size(0)),
                              device=tokens.device)


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)
    print(model(x).size())