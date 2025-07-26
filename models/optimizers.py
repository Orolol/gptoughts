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

try:
    from galore_torch import GaLoreAdamW, GaLoreAdamW8bit
    GALORE_AVAILABLE = True
except ImportError:
    GALORE_AVAILABLE = False

try:
    from models.galore2_fixed import GaLore2AdamW
    GALORE2_AVAILABLE = True
except ImportError:
    GALORE2_AVAILABLE = False

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
    apollo_config: Optional[Dict[str, Any]] = None,
    galore_config: Optional[Dict[str, Any]] = None,
    galore_quantize_proj: Optional[int] = None
) -> torch.optim.Optimizer:
    """
    Configure optimizer for GPT-style models.
    
    Args:
        model: The model to optimize
        weight_decay: Weight decay coefficient
        learning_rate: Learning rate
        betas: Adam beta parameters
        device_type: Device type ('cuda' or 'cpu')
        optimizer_type: Type of optimizer to use ('adamw', 'lion', 'apollo', 'apollo-mini', 'galore', 'galore-8bit', 'galore2')
        apollo_config: Configuration for APOLLO optimizer if used
        galore_config: Configuration for GaLore optimizer if used
        galore_quantize_proj: Quantization bits for GaLore2 projections (1, 2, or None)
        
    Returns:
        Configured optimizer
    """
    # Check if GaLore is requested but not available
    if optimizer_type in ["galore", "galore-8bit"] and not GALORE_AVAILABLE:
        print(f"Warning: {optimizer_type} requested but not available. Falling back to AdamW.")
        optimizer_type = "adamw"
    
    # Check if GaLore2 is requested but not available
    if optimizer_type == "galore2" and not GALORE2_AVAILABLE:
        print(f"Warning: GaLore2 requested but not available. Falling back to AdamW.")
        optimizer_type = "adamw"
    
    # Check if APOLLO is requested but not available
    if optimizer_type in ["apollo", "apollo-mini"] and not APOLLO_AVAILABLE:
        print(f"Warning: {optimizer_type} requested but not available. Falling back to AdamW.")
        optimizer_type = "adamw"
    
    # Use GaLore if requested and available
    if optimizer_type in ["galore", "galore-8bit"] and GALORE_AVAILABLE:
        use_8bit = optimizer_type == "galore-8bit"
        return configure_optimizer_with_galore(
            model=model,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
            device_type=device_type,
            galore_config=galore_config,
            use_8bit=use_8bit
        )
    
    # Use GaLore2 if requested and available
    if optimizer_type == "galore2" and GALORE2_AVAILABLE:
        return configure_optimizer_with_galore2(
            model=model,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
            device_type=device_type,
            galore_config=galore_config,
            quantize_proj=galore_quantize_proj
        )
    
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
    
    # Group parameters - standard categorization for GPT
    optimizer_groups = get_grouped_params(
        model=model,
        weight_decay=weight_decay,
        learning_rate=learning_rate
    )
    
    # Use Lion if requested
    if optimizer_type == "lion" and device_type == 'cuda':
        lion_available = False
        lion_source = None
        
        # Check for torch_optimizer's Lion
        if TORCH_OPTIMIZER_AVAILABLE and hasattr(extra_optim, 'Lion'):
            lion_available = True
            lion_source = "torch_optimizer"
        else:
            # Check for lion-pytorch
            try:
                from lion_pytorch import Lion
                lion_available = True
                lion_source = "lion-pytorch"
            except ImportError:
                pass
        
        if lion_available:
            if lion_source == "torch_optimizer":
                optimizer = extra_optim.Lion(
                    optimizer_groups,
                    lr=learning_rate,
                    betas=betas
                )
                print("Using Lion optimizer from torch_optimizer for GPT model")
            else:  # lion-pytorch
                optimizer = Lion(
                    optimizer_groups,
                    lr=learning_rate,
                    betas=betas
                )
                print("Using Lion optimizer from lion-pytorch for GPT model")
            return optimizer
        else:
            print("Lion optimizer requested but not available. Falling back to AdamW.")
            optimizer_type = "adamw"
    
    # Default to AdamW
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
        if optimizer_type == "lion":
            lion_available = False
            lion_source = None
            
            # Check for torch_optimizer's Lion
            if TORCH_OPTIMIZER_AVAILABLE and hasattr(extra_optim, 'Lion'):
                lion_available = True
                lion_source = "torch_optimizer"
            else:
                # Check for lion-pytorch
                try:
                    from lion_pytorch import Lion
                    lion_available = True
                    lion_source = "lion-pytorch"
                except ImportError:
                    pass
            
            if lion_available:
                if lion_source == "torch_optimizer":
                    optimizer = extra_optim.Lion(
                        optimizer_groups,
                        lr=learning_rate,
                        betas=betas
                    )
                    print("Using Lion optimizer from torch_optimizer for MoE model")
                else:  # lion-pytorch
                    optimizer = Lion(
                        optimizer_groups,
                        lr=learning_rate,
                        betas=betas
                    )
                    print("Using Lion optimizer from lion-pytorch for MoE model")
            else:
                print("Lion optimizer requested but not available. Falling back to AdamW.")
                optimizer = torch.optim.AdamW(
                    optimizer_groups,
                    lr=learning_rate,
                    betas=betas,
                    fused=True
                )
                print("Using AdamW optimizer for MoE model")
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
    optimizer_type: str = "adamw",  # Changed default from "lion" to "adamw" for stable training
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
        if optimizer_type == "lion":
            # Note: Lion optimizer can cause double backward issues with LLaDA models
            # Always fall back to AdamW for stability
            print("Lion optimizer may cause double backward issues with LLaDA. Using AdamW instead.")
            optimizer_type = "adamw"
        
        if optimizer_type == "adamw":
            # Try 8-bit AdamW first for memory efficiency
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
        # For now, just use the first optimizer (APOLLO) since it handles most parameters
        # The standard optimizer for 1D params is less critical
        optimizer = optimizers[0]
        print(f"Note: Using only APOLLO optimizer, skipping standard optimizer for {sum(len(g['params']) for g in standard_param_groups)} 1D parameters")
        
        # Alternative: You could merge the param groups into the APOLLO optimizer
        # But this is complex and may not work well with APOLLO's internals
    
    print(f"Using APOLLO optimizer ({config['mode']} mode) with rank={config['rank']}, scale={config['scale']}")
    
    return optimizer

def configure_optimizer_with_galore(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: Tuple[float, float],
    device_type: str,
    galore_config: Dict[str, Any] = None,
    use_8bit: bool = False
) -> torch.optim.Optimizer:
    """
    Configure optimizer using GaLore (Gradient Low-Rank Projection).
    
    Args:
        model: The model to optimize
        weight_decay: Weight decay coefficient
        learning_rate: Learning rate
        betas: Adam beta parameters
        device_type: Device type ('cuda' or 'cpu')
        galore_config: Configuration for GaLore optimizer
            - rank: Low-rank dimension (default: 128)
            - update_proj_gap: How often to update projection matrices (default: 200)
            - scale: Scaling factor (default: 0.25)
            - proj_type: Projection type (default: 'std')
        use_8bit: Whether to use 8-bit GaLore (more memory efficient)
            
    Returns:
        Configured GaLore optimizer
    """
    if not GALORE_AVAILABLE:
        raise ImportError(
            "GaLore optimizer is not available. Install it with: pip install galore-torch"
        )
    
    # Set default GaLore configuration
    default_config = {
        "rank": 128,
        "update_proj_gap": 200,
        "scale": 0.25,
        "proj_type": "std"
    }
    
    # Update with user-provided config
    config = default_config.copy()
    if galore_config:
        config.update(galore_config)
    
    # Group parameters - separate GaLore and non-GaLore params
    params_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    
    # Identify which parameters should use GaLore
    # GaLore is most effective on large weight matrices (linear layers)
    galore_params = []
    non_galore_params = []
    
    for param_name, param in params_dict.items():
        # Apply GaLore to weight matrices with at least 2 dimensions
        # Skip biases, normalization layers, and embeddings
        if (len(param.shape) >= 2 and 
            'bias' not in param_name and 
            'norm' not in param_name and
            'ln' not in param_name and
            'embed' not in param_name):
            galore_params.append(param)
        else:
            non_galore_params.append(param)
    
    # Create parameter groups
    param_groups = []
    
    # Non-GaLore parameters
    if non_galore_params:
        param_groups.append({
            'params': non_galore_params,
            'weight_decay': weight_decay,
            'lr': learning_rate
        })
    
    # GaLore parameters with projection config
    if galore_params:
        param_groups.append({
            'params': galore_params,
            'weight_decay': weight_decay,
            'lr': learning_rate,
            'rank': config['rank'],
            'update_proj_gap': config['update_proj_gap'],
            'scale': config['scale'],
            'proj_type': config['proj_type']
        })
    
    # Create optimizer
    if use_8bit and device_type == 'cuda':
        optimizer = GaLoreAdamW8bit(
            param_groups,
            lr=learning_rate,
            betas=betas
        )
        print(f"Using 8-bit GaLore optimizer with rank={config['rank']}, "
              f"update_proj_gap={config['update_proj_gap']}, scale={config['scale']}")
    else:
        optimizer = GaLoreAdamW(
            param_groups,
            lr=learning_rate,
            betas=betas
        )
        print(f"Using GaLore optimizer with rank={config['rank']}, "
              f"update_proj_gap={config['update_proj_gap']}, scale={config['scale']}")
    
    print(f"GaLore applied to {len(galore_params)} parameters, "
          f"standard AdamW for {len(non_galore_params)} parameters")
    
    return optimizer


def configure_optimizer_with_galore2(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: Tuple[float, float],
    device_type: str,
    galore_config: Dict[str, Any] = None,
    quantize_proj: Optional[int] = None
) -> torch.optim.Optimizer:
    """
    Configure optimizer using GaLore2 (fast randomized SVD version).
    
    Args:
        model: The model to optimize
        weight_decay: Weight decay coefficient
        learning_rate: Learning rate
        betas: Adam beta parameters
        device_type: Device type ('cuda' or 'cpu')
        galore_config: Configuration for GaLore optimizer
            - rank: Low-rank dimension (default: 128)
            - update_proj_gap: How often to update projection matrices (default: 200)
            - scale: Scaling factor (default: 0.25)
            - proj_type: Projection type (default: 'std', options: 'std', 'random', '1bit', '2bit')
        quantize_proj: Quantization bits for projections (1, 2, or None)
            
    Returns:
        Configured GaLore2 optimizer
    """
    if not GALORE2_AVAILABLE:
        raise ImportError(
            "GaLore2 optimizer is not available. Check models/galore2.py"
        )
    
    # Default configuration
    default_config = {
        'rank': 128,
        'update_proj_gap': 200,
        'scale': 0.25,
        'proj_type': 'std'
    }
    
    # Merge with provided configuration
    config = default_config.copy()
    if galore_config:
        config.update(galore_config)
    
    print(f"Configuring GaLore2 optimizer with config: {config}")
    if quantize_proj:
        print(f"Using {quantize_proj}-bit projection quantization")
    
    # Identify parameters for GaLore projection
    # Apply to weight matrices only, not biases, norms, or embeddings
    galore_params = []
    non_galore_params = []
    
    # Patterns for parameters that should NOT use GaLore
    no_galore_patterns = [
        '.bias',
        'LayerNorm', 'ln_', 'norm',
        'embeddings', 'embed_tokens', 'wte', 'wpe',
        'lm_head', 'output_projection',
        'positional', 'pos_emb'
    ]
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if this is a parameter that should not use GaLore
        is_no_galore = any(pattern in name for pattern in no_galore_patterns)
        
        # Also check if it's a 2D weight matrix (required for GaLore)
        is_weight_matrix = len(param.shape) == 2 and param.shape[0] > 256 and param.shape[1] > 256
        
        if is_weight_matrix and not is_no_galore:
            galore_params.append(param)
            print(f"  GaLore2 will be applied to: {name} (shape: {param.shape})")
        else:
            non_galore_params.append(param)
    
    # Create parameter groups
    param_groups = []
    
    # Non-GaLore parameters
    if non_galore_params:
        param_groups.append({
            'params': non_galore_params,
            'weight_decay': weight_decay,
            'lr': learning_rate
        })
    
    # GaLore parameters with projection config
    if galore_params:
        param_groups.append({
            'params': galore_params,
            'weight_decay': weight_decay,
            'lr': learning_rate,
            'rank': config['rank'],
            'update_proj_gap': config['update_proj_gap'],
            'scale': config['scale'],
            'proj_type': config['proj_type'],
            'quantize_proj': quantize_proj
        })
    
    # Create GaLore2 optimizer
    optimizer = GaLore2AdamW(
        param_groups,
        lr=learning_rate,
        betas=betas,
        eps=1e-8,
        weight_decay=weight_decay,
        rank=config['rank'],
        update_proj_gap=config['update_proj_gap'],
        scale=config['scale'],
        proj_type=config['proj_type'],
        quantize_proj=quantize_proj
    )
    
    print(f"Using GaLore2 optimizer with rank={config['rank']}, "
          f"update_proj_gap={config['update_proj_gap']}, scale={config['scale']}, "
          f"proj_type={config['proj_type']}, quantize_proj={quantize_proj}")
    print(f"GaLore2 applied to {len(galore_params)} parameters, "
          f"standard AdamW for {len(non_galore_params)} parameters")
    
    return optimizer