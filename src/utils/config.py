from dataclasses import dataclass, field
import torch
import pathlib

DATASET_BASE_DIR = pathlib.Path(__file__).parent.parent.parent / 'datasets'

CUDA_SAFE_BATCH_SIZE = 4 # NOTE: This is the maximum batch size that can be used on my 3070 GPU (This value is hardcoded for now but should be made dynamic in the future)

def get_device():
    """
    Get the device to use for training (cuda if available, then mps, then cpu)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

@dataclass
class Hyperparameters:
    optimizer: str 
    loss_fn: str
    batch_size: int = 1
    learning_rate: float = 0.01
    n_epochs: int = 10
    accumulation_steps: int = 1 # Gradient accumulation
    additional_params: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.batch_size > 0, "Batch size must be greater than 0"
        assert self.learning_rate > 0, "Learning rate must be greater than 0"
        assert self.n_epochs > 0, "Number of epochs must be greater than 0"

    def to_dict(self):
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "n_epochs": self.n_epochs,
            "optimizer": self.optimizer,
            "loss_fn": self.loss_fn,
            'accumulation_steps': self.accumulation_steps,
            **self.additional_params
        }

@dataclass
class TrainingConfig:
    architecture: str
    dataset: str
    n_folds: int = 1
    random_state: int = 42
    device: torch.device = field(default_factory=lambda: get_device())
    dataset_root_dir: str = DATASET_BASE_DIR
    use_wandb: bool = False
    hyperparameters: Hyperparameters = field(default_factory=Hyperparameters)

    def __post_init__(self):
        assert pathlib.Path(self.dataset_root_dir).exists(), f"Dataset root directory {self.dataset_root_dir} does not exist"
        assert self.n_folds > 0, "Number of folds must be greater than 0"

        # check if device is cuda 
        if self.device.type == 'cuda' and self.hyperparameters.batch_size > CUDA_SAFE_BATCH_SIZE:
            assert self.hyperparameters.batch_size % CUDA_SAFE_BATCH_SIZE == 0, "Batch size must be a multiple of 4 for CUDA"
            self.hyperparameters.accumulation_steps = self.hyperparameters.batch_size // CUDA_SAFE_BATCH_SIZE
            self.hyperparameters.batch_size = CUDA_SAFE_BATCH_SIZE


            
