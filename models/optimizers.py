"""Optimizer configurations for different model architectures."""

import torch
from typing import List, Dict, Any, Optional, Tuple

try:
    import torch_optimizer as extra_optim
    TORCH_OPTIMIZER_AVAILABLE = True
except ImportError:
    TORCH_OPTIMIZER_AVAILABLE = False

try:
    from apollo_torch import APOLLOAdamW
    APOLLO_AVAILABLE = True
except ImportError:
    APOLLO_AVAILABLE = False

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
    device_type: str,
    optimizer_type: str = "adamw",
    apollo_config: Optional[Dict[str, Any]] = None
) -> torch.optim.Optimizer:
    """
    Configure optimizer for GPT-style models.
    
    Args:
        model: The model to optimize
        weight_decay: Weight decay coefficient
        learning_rate: Learning rate
        betas: Adam beta parameters
        device_type: Device type ('cuda' or 'cpu')
        optimizer_type: Type of optimizer to use ('adamw', 'apollo', 'apollo-mini')
        apollo_config: Configuration for APOLLO optimizer if used
        
    Returns:
        Configured optimizer
    """
    # Check if APOLLO is requested but not available
    if optimizer_type in ["apollo", "apollo-mini"] and not APOLLO_AVAILABLE:
        print(f"Warning: {optimizer_type} requested but not available. Falling back to AdamW.")
        optimizer_type = "adamw"
    
    # Use APOLLO if requested and available
    if optimizer_type in ["apollo", "apollo-mini"] and APOLLO_AVAILABLE:
        # Set default APOLLO configuration based on requested type
        default_config = {"mode": optimizer_type}
        
        # Merge with user-provided config
        config = default_config.copy()
        if apollo_config:
            config.update(apollo_config)
            
        return configure_optimizer_with_apollo(
            model=model,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
            device_type=device_type,
            apollo_config=config
        )
    
    # Otherwise, use standard AdamW
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
    device_type: str,
    optimizer_type: str = "lion",
    apollo_config: Optional[Dict[str, Any]] = None
) -> torch.optim.Optimizer:
    """
    Configure optimizer for MoE models with specialized parameter groups.
    
    Args:
        model: The model to optimize
        weight_decay: Weight decay coefficient
        learning_rate: Learning rate
        betas: Adam beta parameters
        device_type: Device type ('cuda' or 'cpu')
        optimizer_type: Type of optimizer to use ('lion', 'adamw', 'apollo', 'apollo-mini')
        apollo_config: Configuration for APOLLO optimizer if used
        
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
    
    # Check if APOLLO is requested but not available
    if optimizer_type in ["apollo", "apollo-mini"] and not APOLLO_AVAILABLE:
        print(f"Warning: {optimizer_type} requested but not available. Falling back to Lion/AdamW.")
        optimizer_type = "lion" if TORCH_OPTIMIZER_AVAILABLE else "adamw"
    
    # Use APOLLO if requested and available
    if optimizer_type in ["apollo", "apollo-mini"] and APOLLO_AVAILABLE:
        # Set default APOLLO configuration based on requested type
        default_config = {"mode": optimizer_type}
        
        # Merge with user-provided config
        config = default_config.copy()
        if apollo_config:
            config.update(apollo_config)
            
        return configure_optimizer_with_apollo(
            model=model,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
            device_type=device_type,
            apollo_config=config
        )
    
    # Choose optimizer based on device type and availability
    if device_type == 'cuda':
        if optimizer_type == "lion" and TORCH_OPTIMIZER_AVAILABLE:
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
    device_type: str,
    optimizer_type: str = "lion",
    apollo_config: Optional[Dict[str, Any]] = None
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
    
    # Check if APOLLO is requested but not available
    if optimizer_type in ["apollo", "apollo-mini"] and not APOLLO_AVAILABLE:
        print(f"Warning: {optimizer_type} requested but not available. Falling back to Lion/AdamW.")
        optimizer_type = "lion" if TORCH_OPTIMIZER_AVAILABLE else "adamw"
    
    # Use APOLLO if requested and available
    if optimizer_type in ["apollo", "apollo-mini"] and APOLLO_AVAILABLE:
        # Set default APOLLO configuration based on requested type
        default_config = {"mode": optimizer_type}
        
        # Merge with user-provided config
        config = default_config.copy()
        if apollo_config:
            config.update(apollo_config)
            
        return configure_optimizer_with_apollo(
            model=model,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
            device_type=device_type,
            apollo_config=config
        )
    
    # Choose optimizer based on device type and availability
    if device_type == 'cuda':
        if optimizer_type == "lion" and TORCH_OPTIMIZER_AVAILABLE:
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

def configure_optimizer_with_apollo(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: Tuple[float, float],
    device_type: str,
    apollo_config: Dict[str, Any] = None
) -> torch.optim.Optimizer:
    """
    Configure optimizer using APOLLO, a memory-efficient optimizer for LLM training.
    
    Args:
        model: The model to optimize
        weight_decay: Weight decay coefficient
        learning_rate: Learning rate
        betas: Adam beta parameters
        device_type: Device type ('cuda' or 'cpu')
        apollo_config: Configuration for APOLLO optimizer
            - mode: 'apollo' or 'apollo-mini'
            - rank: Rank of auxiliary subspace (default: 256 for apollo, 1 for apollo-mini)
            - scale: Scaling factor (default: 1 for apollo, 128 for apollo-mini)
            - update_proj_gap: Interval for projection updates (default: 200)
            
    Returns:
        Configured APOLLO optimizer
    """
    if not APOLLO_AVAILABLE:
        raise ImportError(
            "APOLLO optimizer is not available. Install it with: pip install apollo-torch"
        )
    
    # Set default APOLLO configuration
    default_config = {
        "mode": "apollo",  # 'apollo' or 'apollo-mini'
        "rank": None,      # Will be set based on mode
        "scale": None,     # Will be set based on mode
        "update_proj_gap": 200,
        "proj": "random",
        "proj_type": "std"
    }
    
    # Update with user-provided config
    config = default_config.copy()
    if apollo_config:
        config.update(apollo_config)
    
    # Set defaults based on mode
    if config["rank"] is None:
        config["rank"] = 256 if config["mode"] == "apollo" else 1
    
    if config["scale"] is None:
        config["scale"] = 1 if config["mode"] == "apollo" else 128
    
    # Set scale_type based on mode
    config["scale_type"] = "channel" if config["mode"] == "apollo" else "tensor"
    
    # Group parameters - standard categorization
    optimizer_groups = get_grouped_params(
        model=model,
        weight_decay=weight_decay,
        learning_rate=learning_rate
    )
    
    # Separate parameters based on their dimensionality
    apollo_param_groups = []
    standard_param_groups = []
    
    for group in optimizer_groups:
        # Create new parameter lists
        apollo_params = []
        standard_params = []
        
        # Check each parameter's dimensionality
        for param in group['params']:
            if len(param.shape) >= 2:  # Parameter has at least 2 dimensions
                apollo_params.append(param)
            else:  # Parameter has only 1 dimension (vector) or 0 dimensions (scalar)
                standard_params.append(param)
        
        # Create APOLLO group if there are eligible parameters
        if apollo_params:
            apollo_group = {
                'params': apollo_params,
                'weight_decay': group.get('weight_decay', 0.0),
                'lr': group.get('lr', learning_rate),
                'rank': config["rank"],
                'proj': config["proj"],
                'scale_type': config["scale_type"],
                'scale': config["scale"],
                'update_proj_gap': config["update_proj_gap"],
                'proj_type': config["proj_type"]
            }
            apollo_param_groups.append(apollo_group)
        
        # Create standard group if there are 1D parameters
        if standard_params:
            standard_group = {
                'params': standard_params,
                'weight_decay': group.get('weight_decay', 0.0),
                'lr': group.get('lr', learning_rate)
            }
            standard_param_groups.append(standard_group)
    
    # Create optimizers
    optimizers = []
    
    # Create APOLLO optimizer if there are eligible parameters
    if apollo_param_groups:
        apollo_opt = APOLLOAdamW(
            apollo_param_groups,
            lr=learning_rate,
            betas=betas
        )
        optimizers.append(apollo_opt)
        print(f"Using APOLLO optimizer ({config['mode']} mode) with rank={config['rank']}, scale={config['scale']} for {sum(len(g['params']) for g in apollo_param_groups)} parameters")
    
    # Create standard optimizer for 1D parameters
    if standard_param_groups:
        if device_type == 'cuda':
            standard_opt = torch.optim.AdamW(
                standard_param_groups,
                lr=learning_rate,
                betas=betas,
                fused=True
            )
        else:
            standard_opt = torch.optim.AdamW(
                standard_param_groups,
                lr=learning_rate,
                betas=betas
            )
        optimizers.append(standard_opt)
        print(f"Using standard AdamW for {sum(len(g['params']) for g in standard_param_groups)} 1D parameters")
    
    # Create a combined optimizer if needed
    if len(optimizers) == 1:
        optimizer = optimizers[0]
    else:
        # Use a simple wrapper class that delegates to multiple optimizers
        class MultiOptimizer:
            def __init__(self, optimizers):
                self.optimizers = optimizers
                # Store combined param_groups for compatibility
                self._param_groups = []
                for opt in optimizers:
                    self._param_groups.extend(opt.param_groups)
            
            def zero_grad(self, set_to_none=False):
                for optimizer in self.optimizers:
                    optimizer.zero_grad(set_to_none=set_to_none)
            
            def step(self, closure=None):
                loss = None
                if closure is not None:
                    loss = closure()
                
                for optimizer in self.optimizers:
                    optimizer.step()
                
                return loss
            
            @property
            def param_groups(self):
                # Refresh combined param_groups
                self._param_groups = []
                for opt in self.optimizers:
                    self._param_groups.extend(opt.param_groups)
                return self._param_groups
            
            def state_dict(self):
                # Combine state dicts from all optimizers
                return {f"opt_{i}": opt.state_dict() for i, opt in enumerate(self.optimizers)}
            
            def load_state_dict(self, state_dict):
                # Load state for each optimizer
                for i, opt in enumerate(self.optimizers):
                    if f"opt_{i}" in state_dict:
                        opt.load_state_dict(state_dict[f"opt_{i}"])
        
        optimizer = MultiOptimizer(optimizers)
        print(f"Using MultiOptimizer with {len(optimizers)} sub-optimizers")
    
    print(f"Using APOLLO optimizer ({config['mode']} mode) with rank={config['rank']}, scale={config['scale']}")
    
    return optimizer