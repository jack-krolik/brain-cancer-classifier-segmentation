from dataclasses import dataclass, field
import torch
import pathlib

DATASET_BASE_DIR = pathlib.Path(__file__).parent.parent.parent / 'datasets'

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
    disable_wandb: bool = False
    hyperparameters: Hyperparameters = field(default_factory=Hyperparameters)

    def __post_init__(self):
        assert pathlib.Path(self.dataset_root_dir).exists(), f"Dataset root directory {self.dataset_root_dir} does not exist"
        assert self.n_folds > 0, "Number of folds must be greater than 0"

            
