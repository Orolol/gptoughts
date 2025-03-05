"""Optimizer configurations for different model architectures."""

import torch
from typing import List, Dict, Any, Optional, Tuple

try:
    import torch_optimizer as extra_optim
    TORCH_OPTIMIZER_AVAILABLE = True
except ImportError:
    TORCH_OPTIMIZER_AVAILABLE = False

def get_grouped_params(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: Optional[float] = None,
    no_decay_names: Optional[List[str]] = None,
    lr_scale: Optional[Dict[str, float]] = None,
    custom_param_groups: Optional[Dict[str, Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Group model parameters for optimizers with specific configurations per group.
    
    Args:
        model: Model to optimize
        weight_decay: Weight decay coefficient
        learning_rate: Base learning rate (optional)
        no_decay_names: List of parameter name patterns to exclude from weight decay
        lr_scale: Dictionary mapping parameter name patterns to learning rate scale factors
        custom_param_groups: Custom parameter groups by name pattern
        
    Returns:
        List of parameter groups for optimizer
    """
    # Default patterns for parameters that don't get weight decay
    if no_decay_names is None:
        no_decay_names = ['.bias', 'LayerNorm.weight', 'LayerNorm.bias', 'ln_', 'norm', 'embeddings']
    
    # Default param groups
    params_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    
    # Initialize parameter groups
    decay_params = []
    no_decay_params = []
    
    # Custom parameter groups by pattern
    custom_groups = {}
    if custom_param_groups:
        for pattern, config in custom_param_groups.items():
            custom_groups[pattern] = {"params": [], **config}
    
    # Categorize parameters
    for param_name, param in params_dict.items():
        # Skip if not requires grad
        if not param.requires_grad:
            continue
        
        # First check if param belongs to a custom group
        found_custom = False
        if custom_param_groups:
            for pattern, group in custom_groups.items():
                if pattern in param_name:
                    group["params"].append(param)
                    found_custom = True
                    break
        
        if found_custom:
            continue
            
        # Otherwise, standard decay/no_decay split
        is_no_decay = any(nd in param_name for nd in no_decay_names)
        
        if is_no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    # Create optimizer param groups
    optimizer_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    # Add custom parameter groups
    if custom_param_groups:
        for group in custom_groups.values():
            if group["params"]:  # Only add if there are parameters
                optimizer_groups.append(group)
    
    # Apply learning rate scaling if provided
    if learning_rate is not None and lr_scale is not None:
        # Apply learning rate scaling to custom groups
        for param_group in optimizer_groups:
            param_group["lr"] = learning_rate
        
        # Check additional groups for learning rate scaling
        for pattern, scale in lr_scale.items():
            for group in optimizer_groups:
                if pattern in str(group.get("name", "")):
                    group["lr"] = learning_rate * scale
    
    return optimizer_groups

def configure_optimizer_for_gpt(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: Tuple[float, float],
    device_type: str
) -> torch.optim.Optimizer:
    """
    Configure optimizer for GPT-style models.
    
    Args:
        model: The model to optimize
        weight_decay: Weight decay coefficient
        learning_rate: Learning rate
        betas: Adam beta parameters
        device_type: Device type ('cuda' or 'cpu')
        
    Returns:
        Configured optimizer
    """
    # Group parameters - standard categorization for GPT
    optimizer_groups = get_grouped_params(
        model=model,
        weight_decay=weight_decay,
        learning_rate=learning_rate
    )
    
    # Create optimizer based on device type
    if device_type == 'cuda':
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=learning_rate,
            betas=betas,
            fused=True  # Use fused implementation for better performance
        )
    else:
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=learning_rate,
            betas=betas
        )
    
    return optimizer

def configure_optimizer_for_moe(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: Tuple[float, float],
    device_type: str
) -> torch.optim.Optimizer:
    """
    Configure optimizer for MoE models with specialized parameter groups.
    
    Args:
        model: The model to optimize
        weight_decay: Weight decay coefficient
        learning_rate: Learning rate
        betas: Adam beta parameters
        device_type: Device type ('cuda' or 'cpu')
        
    Returns:
        Configured optimizer
    """
    # Define custom parameter groups for MoE
    custom_param_groups = {
        "router": {
            "name": "router_params",
            "weight_decay": weight_decay * 0.5,  # Lower weight decay for router
            "lr": learning_rate * 0.1           # Lower learning rate for stability
        },
        "expert": {
            "name": "expert_params",
            "weight_decay": weight_decay,        # Normal weight decay for experts
            "lr": learning_rate                  # Normal learning rate
        }
    }
    
    # Group parameters with MoE-specific categorization
    optimizer_groups = get_grouped_params(
        model=model,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        custom_param_groups=custom_param_groups
    )
    
    # Choose optimizer based on device type and availability
    if device_type == 'cuda':
        if TORCH_OPTIMIZER_AVAILABLE:
            # Lion optimizer often works better for MoE
            optimizer = extra_optim.Lion(
                optimizer_groups,
                lr=learning_rate,
                betas=betas
            )
            print("Using Lion optimizer for MoE model")
        else:
            optimizer = torch.optim.AdamW(
                optimizer_groups,
                lr=learning_rate,
                betas=betas,
                fused=True
            )
            print("Using AdamW optimizer for MoE model")
    else:
        # Standard Adam for CPU
        optimizer = torch.optim.Adam(
            optimizer_groups,
            lr=learning_rate,
            betas=betas
        )
        print("Using Adam optimizer for MoE model (CPU)")
    
    return optimizer

def configure_optimizer_for_llada(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: Tuple[float, float],
    device_type: str
) -> torch.optim.Optimizer:
    """
    Configure optimizer for LLaDA model.
    
    Args:
        model: The model to optimize
        weight_decay: Weight decay coefficient
        learning_rate: Learning rate
        betas: Adam beta parameters
        device_type: Device type ('cuda' or 'cpu')
        
    Returns:
        Configured optimizer
    """
    # LLaDA combines MoE and diffusion approach
    # Define custom parameter groups
    custom_param_groups = {
        "router": {
            "name": "router_params",
            "weight_decay": weight_decay * 0.5,  # Lower weight decay for router
            "lr": learning_rate * 0.1           # Lower learning rate for stability
        },
        "expert": {
            "name": "expert_params",
            "weight_decay": weight_decay,        # Normal weight decay for experts
            "lr": learning_rate                  # Normal learning rate
        }
    }
    
    # Group parameters with LLaDA-specific categorization
    optimizer_groups = get_grouped_params(
        model=model,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        custom_param_groups=custom_param_groups,
        # LLaDA uses the same LR for all MLP but different for routers
        no_decay_names=['.bias', 'LayerNorm.weight', 'LayerNorm.bias', 'ln_', 'norm', 
                      'embeddings', 'temperature', 'pos_emb', 'tok_emb']
    )
    
    # Choose optimizer based on device type and availability
    if device_type == 'cuda':
        if TORCH_OPTIMIZER_AVAILABLE:
            # Lion optimizer often works better for diffusion models
            optimizer = extra_optim.Lion(
                optimizer_groups,
                lr=learning_rate,
                betas=betas,
                weight_decay=0.0  # We apply weight decay in param groups
            )
            print("Using Lion optimizer for LLaDA model")
        else:
            # AdamW with 8-bit precision if available
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    optimizer_groups,
                    lr=learning_rate,
                    betas=betas
                )
                print("Using 8-bit AdamW for LLaDA model")
            except ImportError:
                optimizer = torch.optim.AdamW(
                    optimizer_groups,
                    lr=learning_rate,
                    betas=betas,
                    fused=True
                )
                print("Using AdamW for LLaDA model")
    else:
        # Standard Adam for CPU
        optimizer = torch.optim.Adam(
            optimizer_groups,
            lr=learning_rate,
            betas=betas
        )
        print("Using Adam optimizer for LLaDA model (CPU)")
    
    return optimizer 