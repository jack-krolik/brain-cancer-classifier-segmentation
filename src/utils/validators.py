import torch
from functools import wraps

def validate_model_input(checks):
    def decorator(func):
        @wraps(func)
        def wrapper(self, x, *args, **kwargs):
            if not isinstance(x, torch.Tensor):
                raise TypeError("Input must be a torch.Tensor")
            if 'shape' in checks and x.shape[1:] != checks['shape']:
                raise ValueError(f"Input tensor must have shape {checks['shape']}, but got {x.shape[1:]}") # this ignores the batch size
            if 'dims' in checks and x.ndim != checks['dims']:
                raise ValueError(f"Expected input to have {checks['dims']} dimensions, but got {x.ndim}")
            if 'dtype' in checks and x.dtype != checks['dtype']:
                raise TypeError(f"Expected input dtype {checks['dtype']}, but got {x.dtype}")
            if 'device' in checks and x.device != next(self.parameters()).device:
                raise ValueError(f"Input tensor is on {x.device}, but model parameters are on {next(self.parameters()).device}")
            return func(self, x, *args, **kwargs)
        return wrapper
    return decorator
