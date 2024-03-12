import torch
from functools import wraps
from typing import Union, List

def validate_model_input(checks):
    def decorator(func):
        @wraps(func)
        def wrapper(self, x, *args, **kwargs):
            if not isinstance(x, torch.Tensor):
                raise TypeError("Input must be a torch.Tensor")
            if 'shape' in checks:
                validate_shape(x, checks['shape'])
            if 'dims' in checks:
                validate_dims(x, checks['dims'])
            if 'dtype' in checks and x.dtype != checks['dtype']:
                raise TypeError(f"Expected input dtype {checks['dtype']}, but got {x.dtype}")
            if 'device' in checks and x.device != next(self.parameters()).device:
                raise ValueError(f"Input tensor is on {x.device}, but model parameters are on {next(self.parameters()).device}")
            return func(self, x, *args, **kwargs)
        return wrapper
    return decorator


def validate_shape(x: torch.Tensor, shape: tuple):
    """
    Validate that the input tensor has the expected shape

    If the input tensor has a batch dimension, it is ignored (assumes 4D tensor with batch dimension)
    """
    if x.shape != shape and not(len(shape) + 1 == x.ndim and shape == x.shape[1:]):
        raise ValueError(f"Input tensor must have shape {shape}, but got {x.shape[1:]}")
        
def validate_dims(x: torch.Tensor, dims: Union[int, List]):
    """
    Validate that the input tensor has the expected number of dimensions
    """
    if x.ndim == dims:
        return
    if isinstance(dims, list) and x.ndim in dims:
        return
    raise ValueError(f"Input tensor must have {dims} dimensions, but got {x.ndim}")