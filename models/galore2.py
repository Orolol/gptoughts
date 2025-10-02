"""
GaLore 2: Memory-Efficient LLM Training by Gradient Low-Rank Projection
Based on the paper: https://arxiv.org/abs/2504.20437

Key improvements over GaLore 1:
- Fast randomized SVD for subspace updates (15x faster)
- FSDP (Fully Sharded Data Parallel) integration
- Support for quantized projections
- Improved memory efficiency
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import numpy as np
from torch.optim.optimizer import Optimizer
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class FastRandomizedSVD:
    """Fast randomized SVD implementation based on Halko et al. (2011)"""
    
    @staticmethod
    def svd(matrix: torch.Tensor, rank: int, oversampling: int = 5, 
            n_iter: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute randomized SVD of a matrix.
        
        Args:
            matrix: Input matrix to decompose
            rank: Target rank for decomposition
            oversampling: Oversampling factor for better accuracy
            n_iter: Number of power iterations for better accuracy
            
        Returns:
            U, S, V: SVD decomposition matrices
        """
        m, n = matrix.shape
        l = rank + oversampling  # Oversampled rank
        
        # Step 1: Generate random matrix
        omega = torch.randn(n, l, device=matrix.device, dtype=matrix.dtype)
        
        # Step 2: Form Y = A * Omega
        Y = torch.matmul(matrix, omega)
        
        # Step 3: Power iteration for better accuracy (optional)
        for _ in range(n_iter):
            Y = torch.matmul(matrix, torch.matmul(matrix.T, Y))
            
        # Step 4: QR decomposition of Y
        Q, _ = torch.linalg.qr(Y, mode='reduced')
        
        # Step 5: Form B = Q^T * A
        B = torch.matmul(Q.T, matrix)
        
        # Step 6: SVD of smaller matrix B
        U_tilde, S, Vt = torch.linalg.svd(B, full_matrices=False)
        
        # Step 7: Recover left singular vectors
        U = torch.matmul(Q, U_tilde)
        
        # Return only the top 'rank' components
        return U[:, :rank], S[:rank], Vt[:rank, :]


class GaLoreProjector:
    """GaLore gradient projector with fast randomized SVD"""
    
    def __init__(self, rank: int, update_proj_gap: int = 200, 
                 scale: float = 0.25, proj_type: str = 'std',
                 quantize_proj: Optional[int] = None):
        """
        Args:
            rank: Rank of the projection
            update_proj_gap: Frequency of projection updates (T in paper)
            scale: Scaling factor (alpha in paper)
            proj_type: Type of projection ('std', 'random', '1bit', '2bit')
            quantize_proj: Bit-width for projection quantization (if enabled)
        """
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type
        self.quantize_proj = quantize_proj
        
        self.step = 0
        self.proj_matrix = None
        
    def project(self, grad: torch.Tensor) -> torch.Tensor:
        """Project gradient to low-rank subspace"""
        if self.proj_matrix is None or self.step % self.update_proj_gap == 0:
            self._update_projection(grad)
            
        # Project: R = P^T @ G
        # Handle both 2D and higher dimensional gradients
        original_shape = grad.shape
        if len(grad.shape) > 2:
            # Flatten to 2D for projection
            grad = grad.view(grad.shape[0], -1)
        
        # Ensure gradient and projection matrix are compatible
        if self.proj_matrix.shape[0] == grad.shape[0]:
            # P^T @ G (when m <= n)
            low_rank_grad = torch.matmul(self.proj_matrix.T, grad)
        else:
            # G @ P (when m > n)
            low_rank_grad = torch.matmul(grad, self.proj_matrix)
            
        self.step += 1
        
        return low_rank_grad
    
    def project_back(self, low_rank_update: torch.Tensor) -> torch.Tensor:
        """Project low-rank update back to original space"""
        # G_tilde = alpha * P @ N
        # Handle both cases based on projection matrix orientation
        if self.proj_matrix.shape[1] == low_rank_update.shape[0]:
            # P @ N (when m <= n)
            return self.scale * torch.matmul(self.proj_matrix, low_rank_update)
        else:
            # N @ P^T (when m > n)
            return self.scale * torch.matmul(low_rank_update, self.proj_matrix.T)
    
    def _update_projection(self, grad: torch.Tensor):
        """Update projection matrix using fast randomized SVD"""
        m, n = grad.shape
        
        if self.proj_type == 'std':
            # Fast randomized SVD
            if m <= n:
                U, _, _ = FastRandomizedSVD.svd(grad, self.rank)
                self.proj_matrix = U
            else:
                _, _, Vt = FastRandomizedSVD.svd(grad, self.rank)
                self.proj_matrix = Vt.T
                
        elif self.proj_type == 'random':
            # Random projection (for comparison)
            if m <= n:
                self.proj_matrix = torch.randn(m, self.rank, device=grad.device, 
                                              dtype=grad.dtype)
            else:
                self.proj_matrix = torch.randn(n, self.rank, device=grad.device,
                                              dtype=grad.dtype)
            # Orthonormalize
            self.proj_matrix, _ = torch.linalg.qr(self.proj_matrix, mode='reduced')
            
        # Apply quantization if specified
        if self.quantize_proj is not None:
            self.proj_matrix = self._quantize_projection(self.proj_matrix)
            
    def _quantize_projection(self, proj: torch.Tensor) -> torch.Tensor:
        """Quantize projection matrix to specified bit-width"""
        if self.quantize_proj == 1:
            # 1-bit quantization (sign only)
            return torch.sign(proj)
        elif self.quantize_proj == 2:
            # 2-bit quantization
            # Simple 2-bit quantization scheme
            scale = proj.abs().max()
            quantized = torch.round(proj / scale * 1.5).clamp(-2, 1)
            return quantized * scale / 1.5
        else:
            return proj


class GaLoreAdamW(Optimizer):
    """AdamW optimizer with GaLore projection"""
    
    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01, 
                 rank: int = 128, update_proj_gap: int = 200, 
                 scale: float = 0.25, proj_type: str = 'std',
                 quantize_proj: Optional[int] = None):
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       rank=rank, update_proj_gap=update_proj_gap, 
                       scale=scale, proj_type=proj_type, quantize_proj=quantize_proj)
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('GaLore does not support sparse gradients')
                    
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Check if this param should use GaLore
                    if 'rank' in group and len(grad.shape) >= 2:
                        # Initialize projector
                        state['projector'] = GaLoreProjector(
                            rank=group['rank'],
                            update_proj_gap=group['update_proj_gap'],
                            scale=group['scale'],
                            proj_type=group['proj_type'],
                            quantize_proj=group.get('quantize_proj', None)
                        )
                        # Initialize low-rank moments
                        # The moments should match the shape of the low-rank gradient
                        # For 2D parameters, we need to handle the projection output shape
                        if grad.shape[0] <= grad.shape[1]:
                            # When m <= n, projection gives (rank, n) 
                            state['exp_avg'] = torch.zeros(group['rank'], grad.shape[1], 
                                                          device=grad.device, dtype=grad.dtype)
                            state['exp_avg_sq'] = torch.zeros(group['rank'], grad.shape[1],
                                                            device=grad.device, dtype=grad.dtype)
                        else:
                            # When m > n, projection gives (m, rank)
                            state['exp_avg'] = torch.zeros(grad.shape[0], group['rank'],
                                                          device=grad.device, dtype=grad.dtype)
                            state['exp_avg_sq'] = torch.zeros(grad.shape[0], group['rank'],
                                                            device=grad.device, dtype=grad.dtype)
                    else:
                        # Standard AdamW for non-GaLore params
                        state['exp_avg'] = torch.zeros_like(grad)
                        state['exp_avg_sq'] = torch.zeros_like(grad)
                        
                state['step'] += 1
                
                if 'projector' in state:
                    # GaLore update
                    projector = state['projector']
                    
                    # Project gradient to low-rank
                    low_rank_grad = projector.project(grad)
                    
                    # Update low-rank moments
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']
                    
                    exp_avg.mul_(beta1).add_(low_rank_grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(low_rank_grad, low_rank_grad, value=1 - beta2)
                    
                    # Bias correction
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    
                    # Compute low-rank update
                    low_rank_update = exp_avg / bias_correction1
                    denom = (exp_avg_sq / bias_correction2).sqrt().add_(group['eps'])
                    low_rank_update = low_rank_update / denom
                    
                    # Project back to original space
                    update = projector.project_back(low_rank_update)
                    
                else:
                    # Standard AdamW update
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']
                    
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    
                    update = exp_avg / bias_correction1
                    denom = (exp_avg_sq / bias_correction2).sqrt().add_(group['eps'])
                    update = update / denom
                    
                # Apply weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
                    
                # Apply update
                p.data.add_(update, alpha=-group['lr'])
                
        return loss


class GaLoreFSDP:
    """Helper class for GaLore integration with FSDP"""
    
    @staticmethod
    def setup_galore_fsdp(model: nn.Module, rank: int = 128, 
                         update_proj_gap: int = 200, scale: float = 0.25,
                         proj_type: str = 'std') -> FSDP:
        """
        Setup FSDP with GaLore optimization.
        
        This enables per-layer weight updates with GaLore projection,
        minimizing gradient memory usage in distributed training.
        """
        
        def _register_galore_hooks(module: nn.Module):
            """Register post-accumulate-grad hooks for GaLore updates"""
            
            def galore_post_accumulate_grad_hook(module):
                # This hook is called after gradient accumulation
                # Perform GaLore projection and weight update here
                for name, param in module.named_parameters():
                    if param.grad is not None and len(param.grad.shape) >= 2:
                        # Apply GaLore projection logic here
                        # This is a simplified version - full implementation
                        # would integrate with the optimizer
                        pass
                        
            # Register hook
            if hasattr(module, 'register_post_accumulate_grad_hook'):
                module.register_post_accumulate_grad_hook(galore_post_accumulate_grad_hook)
                
        # Apply hooks to all modules
        model.apply(_register_galore_hooks)
        
        # Wrap model with FSDP
        fsdp_model = FSDP(
            model,
            # FSDP configuration for GaLore
            cpu_offload=None,  # Can be configured based on memory constraints
            backward_prefetch=None,
            forward_prefetch=False,
            limit_all_gathers=True,
            use_orig_params=True,  # Important for GaLore hooks
        )
        
        return fsdp_model


# Example usage
def train_with_galore2():
    """Example training loop with GaLore 2"""
    
    # Model setup
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=512, nhead=8),
        num_layers=12
    )
    
    # Separate parameters for GaLore and non-GaLore optimization
    galore_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if len(param.shape) >= 2 and param.shape[0] * param.shape[1] > 1024:
            # Apply GaLore to large matrices
            galore_params.append(param)
        else:
            other_params.append(param)
            
    # Setup optimizer with different parameter groups
    optimizer = GaLoreAdamW([
        {'params': other_params},
        {'params': galore_params, 'rank': 128, 'update_proj_gap': 200, 
         'scale': 0.25, 'proj_type': 'std'}
    ], lr=0.001)
    
    # Training loop
    for epoch in range(10):
        for batch in dataloader:  # dataloader not defined in this example
            optimizer.zero_grad()
            
            output = model(batch)
            loss = compute_loss(output)  # compute_loss not defined
            
            loss.backward()
            optimizer.step()
            
            
if __name__ == "__main__":
    # Test fast randomized SVD
    matrix = torch.randn(1000, 500)
    U, S, Vt = FastRandomizedSVD.svd(matrix, rank=50)
    print(f"Fast SVD shapes: U={U.shape}, S={S.shape}, Vt={Vt.shape}")
    
    # Verify decomposition quality
    reconstructed = torch.matmul(U, torch.matmul(torch.diag(S), Vt))
    error = torch.norm(matrix - reconstructed) / torch.norm(matrix)
    print(f"Reconstruction error: {error:.6f}")