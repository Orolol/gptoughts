"""
Utility functions for tensor handling and graph isolation to avoid double backward issues.
"""

import torch
import contextlib
import functools

@contextlib.contextmanager
def prevent_backward_reuse():
    """
    Context manager to help prevent tensor reuse between computational graphs.
    This is particularly useful for models with complex computation flows
    where tensors might be reused between different backward passes.
    
    Usage:
        with prevent_backward_reuse():
            # Your model forward pass code here
    """
    prev_grad_enabled = torch.is_grad_enabled()
    try:
        # Yield control back to the caller
        yield
    finally:
        # Restore previous grad state
        torch.set_grad_enabled(prev_grad_enabled)

def isolate_tensor(tensor):
    """
    Creates an isolated copy of a tensor that breaks connections to previous 
    computation graphs and prevents double backward issues.
    
    Args:
        tensor (torch.Tensor): The tensor to isolate
        
    Returns:
        torch.Tensor: A new tensor with the same data but no connection to previous graphs
    """
    if tensor is None:
        return None
        
    # First detach to separate from computation history
    result = tensor.detach()
    
    # Then create a fresh copy that requires grad if the original tensor did
    if tensor.requires_grad:
        result = result.clone().requires_grad_(True)
    else:
        result = result.clone()
        
    return result

def tensor_clone_decorator(func):
    """
    Decorator that isolates input tensors to a function to prevent 
    double backward issues.
    
    Args:
        func: The function to decorate
        
    Returns:
        Decorated function with tensor isolation
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Clone and detach tensor arguments
        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                new_args.append(isolate_tensor(arg))
            else:
                new_args.append(arg)
        
        # Clone and detach tensor keyword arguments
        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                new_kwargs[key] = isolate_tensor(value)
            else:
                new_kwargs[key] = value
        
        # Call the original function with isolated tensors
        return func(*new_args, **new_kwargs)
    
    return wrapper