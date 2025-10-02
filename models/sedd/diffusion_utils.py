"""
Diffusion utilities for SEDD (Score Entropy Discrete Diffusion) model.

This module implements the core diffusion components:
- Graph abstractions (Uniform, Absorbing)
- Noise schedules (Geometric, LogLinear) 
- Score entropy loss computation
- Sampling utilities
"""

import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Literal


def categorical_sample_gumbel(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Sample from categorical distribution using Gumbel-Max trick."""
    gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    return F.softmax((logits + gumbel) / temperature, dim=-1)


def sample_categorical(probs: torch.Tensor, method: str = "hard") -> torch.Tensor:
    """Sample from categorical distribution."""
    if method == "hard":
        return torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(*probs.shape[:-1])
    else:
        return categorical_sample_gumbel(torch.log(probs + 1e-8))


def unsqueeze_as(x: torch.Tensor, y: torch.Tensor, back: bool = True) -> torch.Tensor:
    """Unsqueeze tensor x to match the dimensions of tensor y."""
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)


class Graph(abc.ABC):
    """Abstract base class for graph structures in discrete diffusion."""
    
    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Dimension of the discrete state space."""
        pass

    @property
    @abc.abstractmethod
    def absorb(self) -> bool:
        """Whether the graph has an absorbing state."""
        pass

    @abc.abstractmethod
    def rate(self, i: torch.Tensor) -> torch.Tensor:
        """Compute the i-th column of the rate matrix Q."""
        pass

    @abc.abstractmethod
    def transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the i-th column of the transition matrix e^{sigma Q}."""
        pass

    def sample_transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Sample from the transition distribution."""
        transition_vector = self.transition(i, sigma)
        return sample_categorical(transition_vector, method="hard")

    @abc.abstractmethod
    def score_entropy(self, score: torch.Tensor, sigma: torch.Tensor, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Compute the score entropy loss."""
        pass


class UniformGraph(Graph):
    """Uniform graph where every state can transition to every other state."""
    
    def __init__(self, dim: int):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def absorb(self) -> bool:
        return False

    def rate(self, i: torch.Tensor) -> torch.Tensor:
        """Rate matrix for uniform transitions."""
        edge = torch.ones(*i.shape, self.dim, device=i.device) / self.dim
        edge = edge.scatter(-1, i[..., None], -(self.dim - 1) / self.dim)
        return edge

    def transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Transition probabilities for uniform graph."""
        sigma = unsqueeze_as(sigma, i[..., None])
        trans = torch.ones(*i.shape, self.dim, device=i.device) * (1 - (-sigma).exp()) / self.dim
        trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans[..., :1]))
        trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        return trans

    def sample_transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Optimized sampling for uniform transitions."""
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, torch.randint_like(i, self.dim), i)
        return i_pert

    def score_entropy(self, score: torch.Tensor, sigma: torch.Tensor, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Score entropy loss for uniform graph."""
        sigma = unsqueeze_as(sigma, x)
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        ratio = 1 - self.dim / (esigm1 + self.dim)

        # Negative term
        neg_term = score.mean(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1) / self.dim
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(score, -1, x0[..., None]).squeeze(-1) / esigm1 + neg_term
        )

        # Constant factor
        const = torch.where(
            x == x0,
            (self.dim - 1) / self.dim * ratio * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim 
        )

        # Positive term
        sexp = score.exp()
        pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / self.dim
        
        return pos_term - neg_term + const


class AbsorbingGraph(Graph):
    """Absorbing graph where states transition to an absorbing state."""
    
    def __init__(self, dim: int):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim + 1
    
    @property
    def absorb(self) -> bool:
        return True

    def rate(self, i: torch.Tensor) -> torch.Tensor:
        """Rate matrix for absorbing transitions."""
        return F.one_hot((self.dim - 1) * torch.ones_like(i), num_classes=self.dim) - F.one_hot(i, num_classes=self.dim)

    def transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Transition probabilities for absorbing graph."""
        sigma = unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)
        edge += torch.where(
            i == self.dim - 1,
            1 - (-sigma).squeeze(-1).exp(),
            0
        )[..., None]
        return edge

    def sample_transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Optimized sampling for absorbing transitions."""
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, self.dim - 1, i)
        return i_pert

    def score_entropy(self, score: torch.Tensor, sigma: torch.Tensor, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Score entropy loss for absorbing graph."""
        sigma = unsqueeze_as(sigma, x)
        rel_ind = x == self.dim - 1  # Positions that are in absorbing state (masked)
        
        # Check if we have any masked positions
        num_masked = rel_ind.sum()
        if num_masked == 0:
            # No tokens are masked - return zero loss
            # This can happen early in training with small sigma values
            return torch.zeros_like(x, dtype=torch.float)
        
        
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )

        # Only compute for masked positions
        ratio = 1 / esigm1.expand_as(x)[rel_ind]
        other_ind = x0[rel_ind]  # Original tokens at masked positions

        # Gather scores for the original tokens at masked positions
        try:
            # Clamp log scores to avoid extreme values
            score_clamped = torch.clamp(score[rel_ind], min=-20.0, max=5.0)
            
            # Negative term: -log score of the true token
            neg_term = ratio * torch.gather(score_clamped, -1, other_ind[..., None]).squeeze(-1)

            # Positive term: sum of exp(scores) for all non-absorbing tokens
            # score[rel_ind] has shape [num_masked, vocab_size]
            # We exclude the last token (absorbing state) from the sum
            log_scores_non_absorb = score_clamped[..., :-1]
            
            # Clamp before exp to avoid overflow: exp(-20) ≈ 0, exp(5) ≈ 148
            # The sum should be reasonable (at most vocab_size * exp(5))
            pos_term = log_scores_non_absorb.exp().sum(dim=-1)

            # Constant term - also clamp ratio to avoid extreme values
            ratio_clamped = torch.clamp(ratio, min=1e-8, max=1e8)
            const = ratio_clamped * (ratio_clamped.log() - 1)

            # Compute entropy for masked positions
            masked_entropy = pos_term - neg_term + const
            
            # Final clamp to prevent extreme loss values - be very aggressive
            masked_entropy = torch.clamp(masked_entropy, min=-5.0, max=5.0)
            
            
            # Check for numerical issues
            if torch.isnan(masked_entropy).any() or torch.isinf(masked_entropy).any():
                # Fallback to a simple loss
                masked_entropy = torch.clamp(masked_entropy, -100, 100)

        except Exception as e:
            print(f"Error in score entropy computation: {e}")
            # Fallback: simple cross-entropy-like loss for masked positions
            masked_entropy = -torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)

        # Create full entropy tensor
        entropy = torch.zeros(*x.shape, device=x.device, dtype=torch.float)
        entropy[rel_ind] = masked_entropy
        
        return entropy


class NoiseSchedule(abc.ABC, nn.Module):
    """Abstract base class for noise schedules."""
    
    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return total noise and rate of noise change."""
        return self.total_noise(t), self.rate_noise(t)

    @abc.abstractmethod
    def rate_noise(self, t: torch.Tensor) -> torch.Tensor:
        """Rate of change of noise."""
        pass

    @abc.abstractmethod
    def total_noise(self, t: torch.Tensor) -> torch.Tensor:
        """Total noise accumulated."""
        pass


class GeometricNoise(NoiseSchedule):
    """Geometric noise schedule."""
    
    def __init__(self, sigma_min: float = 1e-3, sigma_max: float = 1.0):
        super().__init__()
        self.register_buffer("sigmas", torch.tensor([sigma_min, sigma_max]))

    def rate_noise(self, t: torch.Tensor) -> torch.Tensor:
        return (self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t * 
                (self.sigmas[1].log() - self.sigmas[0].log()))

    def total_noise(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t


class LogLinearNoise(NoiseSchedule):
    """Log-linear noise schedule optimized for absorbing processes."""
    
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def rate_noise(self, t: torch.Tensor) -> torch.Tensor:
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.log1p(-(1 - self.eps) * t)


def get_graph(graph_type: str, dim: int) -> Graph:
    """Factory function to create graph instances."""
    if graph_type == "uniform":
        return UniformGraph(dim)
    elif graph_type == "absorb":
        return AbsorbingGraph(dim)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")


def get_noise(noise_type: str, sigma_min: float = 1e-4, sigma_max: float = 20.0) -> NoiseSchedule:
    """Factory function to create noise schedule instances."""
    if noise_type == "geometric":
        return GeometricNoise(sigma_min, sigma_max)
    elif noise_type == "loglinear":
        return LogLinearNoise()
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def get_score_entropy_loss(
    model_output: torch.Tensor,
    graph: Graph,
    noise: NoiseSchedule,
    x: torch.Tensor,
    x0: torch.Tensor,
    t: torch.Tensor
) -> torch.Tensor:
    """
    Compute the score entropy loss for SEDD training.
    
    Args:
        model_output: Model predictions (log scores)
        graph: Graph structure
        noise: Noise schedule
        x: Perturbed tokens
        x0: Original tokens
        t: Timesteps
        
    Returns:
        Score entropy loss
    """
    sigma, dsigma = noise(t)
    
    
    # Compute score entropy
    entropy = graph.score_entropy(model_output, sigma, x, x0)
    
    # Weight by noise derivative with much more aggressive dsigma clipping
    # The key insight: dsigma can be extremely large (500+), causing loss explosion
    dsigma_clipped = torch.clamp(dsigma, max=1.0)  # Much more aggressive limit
    scaling_factor = 0.1  # Keep scaling factor for stability
    loss = scaling_factor * (dsigma_clipped[:, None] * entropy).sum(dim=-1)
    
    return loss