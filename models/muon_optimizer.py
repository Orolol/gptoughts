"""
Muon Optimizer - Momentum Orthogonalized by Newton-Schulz

Based on the paper: "Muon: An optimizer for neural networks"
Implementation inspired by the original paper and community implementations.

Muon uses orthogonalized momentum with Newton-Schulz iterations for improved
training dynamics, particularly beneficial for large language models.
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration for computing G^(-1/2).

    Uses 5th order convergence for fast approximation of the matrix inverse square root.

    Args:
        G: Input matrix (typically G = W @ W.T for weight matrix W)
        steps: Number of Newton-Schulz iterations (default: 5)
        eps: Small epsilon for numerical stability

    Returns:
        Approximation of G^(-1/2)
    """
    assert len(G.shape) == 2, "Input must be a 2D matrix"

    # Normalize by the Frobenius norm
    a, b, c = (3.4445, -4.7750, 2.0315)  # 5th order coefficients
    X = G.bfloat16() / (G.norm() + eps)

    # Initialize identity
    if G.size(0) > G.size(1):
        X = X.T

    # Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T

    return X.to(G.dtype)


class Muon(Optimizer):
    """
    Muon optimizer - Momentum Orthogonalized by Newton-schulz.

    This optimizer applies orthogonalization to momentum using Newton-Schulz iterations,
    which can lead to improved training dynamics and better convergence for LLMs.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.02)
        momentum: Momentum factor (default: 0.95)
        nesterov: Whether to use Nesterov momentum (default: True)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
        adamw_params: List of parameter groups that should use AdamW instead of Muon
        adamw_lr: Learning rate for AdamW parameters (default: 3e-4)
        adamw_betas: Beta parameters for AdamW (default: (0.9, 0.95))
        adamw_wd: Weight decay for AdamW parameters (default: 0.1)
        adamw_eps: Epsilon for AdamW (default: 1e-8)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_params: Optional[List] = None,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple = (0.9, 0.95),
        adamw_wd: float = 0.1,
        adamw_eps: float = 1e-8,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps value: {ns_steps}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_wd=adamw_wd,
            adamw_eps=adamw_eps,
        )
        super().__init__(params, defaults)

        # Store AdamW parameter IDs for quick lookup
        self.adamw_param_ids = set()
        if adamw_params is not None:
            for p in adamw_params:
                self.adamw_param_ids.add(id(p))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Check if this parameter should use AdamW
                if id(p) in self.adamw_param_ids:
                    self._step_adamw(p, grad, group)
                    continue

                # Standard Muon update for 2D parameters
                if len(p.shape) >= 2:
                    self._step_muon(p, grad, group, momentum, nesterov, ns_steps)
                else:
                    # For 1D parameters (biases, norms), use simple momentum
                    self._step_simple_momentum(p, grad, group, momentum, nesterov)

        return loss

    def _step_muon(self, p, grad, group, momentum, nesterov, ns_steps):
        """Muon update with orthogonalized momentum."""
        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            state['momentum_buffer'] = torch.zeros_like(grad)

        buf = state['momentum_buffer']

        # Reshape to 2D for Newton-Schulz
        original_shape = grad.shape
        if len(grad.shape) > 2:
            grad_2d = grad.view(grad.shape[0], -1)
        else:
            grad_2d = grad

        # Compute G = grad @ grad.T or grad.T @ grad depending on shape
        if grad_2d.shape[0] <= grad_2d.shape[1]:
            G = grad_2d @ grad_2d.T
        else:
            G = grad_2d.T @ grad_2d

        # Newton-Schulz iterations to get G^(-1/2)
        G_inv_sqrt = zeropower_via_newtonschulz5(G, steps=ns_steps)

        # Orthogonalize gradient
        if grad_2d.shape[0] <= grad_2d.shape[1]:
            grad_ortho = G_inv_sqrt @ grad_2d
        else:
            grad_ortho = grad_2d @ G_inv_sqrt

        # Reshape back
        grad_ortho = grad_ortho.view(original_shape)

        # Momentum update
        buf.mul_(momentum).add_(grad_ortho)

        if nesterov:
            update = grad_ortho + momentum * buf
        else:
            update = buf

        # Apply update
        p.add_(update, alpha=-group['lr'])

        state['step'] += 1

    def _step_simple_momentum(self, p, grad, group, momentum, nesterov):
        """Simple momentum for 1D parameters."""
        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            state['momentum_buffer'] = torch.zeros_like(grad)

        buf = state['momentum_buffer']
        buf.mul_(momentum).add_(grad)

        if nesterov:
            update = grad + momentum * buf
        else:
            update = buf

        p.add_(update, alpha=-group['lr'])

        state['step'] += 1

    def _step_adamw(self, p, grad, group):
        """AdamW update for specified parameters."""
        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(grad)
            state['exp_avg_sq'] = torch.zeros_like(grad)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['adamw_betas']

        state['step'] += 1

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Bias correction
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['adamw_lr'] / bias_correction1

        # Compute norm for AdamW update
        denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['adamw_eps'])

        # Apply weight decay
        p.mul_(1 - group['adamw_lr'] * group['adamw_wd'])

        # Apply update
        p.addcdiv_(exp_avg, denom, value=-step_size)


def create_muon_optimizer(
    model: torch.nn.Module,
    lr: float = 0.02,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    adamw_lr: float = 3e-4,
    adamw_betas: tuple = (0.9, 0.95),
    adamw_wd: float = 0.1,
    adamw_eps: float = 1e-8,
):
    """
    Create Muon optimizer with automatic parameter grouping.

    Large 2D parameters use Muon, while embeddings, norms, and biases use AdamW.

    Args:
        model: Model to optimize
        lr: Learning rate for Muon
        momentum: Momentum factor for Muon
        nesterov: Whether to use Nesterov momentum
        ns_steps: Number of Newton-Schulz iterations
        adamw_lr: Learning rate for AdamW parameters
        adamw_betas: Beta parameters for AdamW
        adamw_wd: Weight decay for AdamW
        adamw_eps: Epsilon for AdamW

    Returns:
        Muon optimizer instance
    """
    # Separate parameters for Muon vs AdamW
    muon_params = []
    adamw_params = []

    # Patterns for parameters that should use AdamW
    adamw_patterns = [
        'embed', 'wte', 'wpe',  # Embeddings
        'norm', 'ln_', 'ln_f',  # Normalization
        'bias',  # Biases
        'lm_head', 'output_projection',  # Output head
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if this parameter should use AdamW
        use_adamw = any(pattern in name.lower() for pattern in adamw_patterns)

        # Also use AdamW for 1D parameters
        use_adamw = use_adamw or len(param.shape) < 2

        if use_adamw:
            adamw_params.append(param)
        else:
            muon_params.append(param)

    print(f"Muon optimizer: {len(muon_params)} parameters with Muon, {len(adamw_params)} with AdamW")

    # Create single parameter group with all parameters
    all_params = muon_params + adamw_params

    optimizer = Muon(
        all_params,
        lr=lr,
        momentum=momentum,
        nesterov=nesterov,
        ns_steps=ns_steps,
        adamw_params=adamw_params,
        adamw_lr=adamw_lr,
        adamw_betas=adamw_betas,
        adamw_wd=adamw_wd,
        adamw_eps=adamw_eps,
    )

    return optimizer
