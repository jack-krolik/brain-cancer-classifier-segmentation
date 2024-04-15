from dataclasses import dataclass, field
import torch
import pathlib
from enum import auto, StrEnum

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
    optimizer: "Optimizer" 
    loss_fn: str
    scheduler: str = None
    batch_size: int = 1
    learning_rate: float = 0.01
    n_epochs: int = 10
    accumulation_steps: int = 1 # Gradient accumulation
    additional_params: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.batch_size > 0, "Batch size must be greater than 0"
        assert self.learning_rate > 0, "Learning rate must be greater than 0"
        assert self.n_epochs > 0, "Number of epochs must be greater than 0"
        assert self.accumulation_steps > 0, "Accumulation steps must be greater than 0"
        assert Optimizer.is_optimizer(self.optimizer), f"Invalid optimizer: {self.optimizer}"
        assert self.scheduler is None or LRScheduler.is_scheduler(self.scheduler), f"Invalid scheduler: {self.scheduler}"

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
    n_checkpoints: int = 1
    checkpoints: list = field(default_factory=list)
    dynamic_batching: bool = True
    hyperparameters: Hyperparameters = field(default_factory=Hyperparameters)

    def __post_init__(self):
        assert pathlib.Path(self.dataset_root_dir).exists(), f"Dataset root directory {self.dataset_root_dir} does not exist"
        assert self.n_folds > 0, "Number of folds must be greater than 0"
        assert self.n_checkpoints > 0 and self.n_checkpoints <= self.hyperparameters.n_epochs, "Number of checkpoints must be greater than 0 and less than or equal to the number of epochs"

        batch_size = self.hyperparameters.batch_size

        # check if device is cuda 
        if self.device.type == 'cuda' and batch_size > CUDA_SAFE_BATCH_SIZE and self.dynamic_batching:
            accumulation_steps = ((batch_size + CUDA_SAFE_BATCH_SIZE - 1) // CUDA_SAFE_BATCH_SIZE)
            print(f"""
            Batch size {batch_size} is too large for the current device. 
            Splitting the batch into {accumulation_steps} steps of {CUDA_SAFE_BATCH_SIZE} samples each.
            """)

            self.hyperparameters.batch_size = CUDA_SAFE_BATCH_SIZE
            self.hyperparameters.accumulation_steps = accumulation_steps
        
        # determine epochs to save model based on number of checkpoints
        save_every = self.hyperparameters.n_epochs // self.n_checkpoints
        checkpoint_epochs = [save_every * i for i in range(1, self.n_checkpoints + 1)]
        checkpoint_epochs[-1] = self.hyperparameters.n_epochs
        self.checkpoints = checkpoint_epochs

    def flatten(self):
        return {
            "architecture": self.architecture,
            "dataset": self.dataset,
            "n_folds": self.n_folds,
            "random_state": self.random_state,
            "device": self.device,
            "dataset_root_dir": self.dataset_root_dir,
            "use_wandb": self.use_wandb,
            **self.hyperparameters.to_dict()
        }
            
class Optimizer(StrEnum):
    SGD = auto()
    ADAM = auto()

    # class method to build optimizer with type, config, and model
    @classmethod
    def build(cls, optimizer_type: str, config: TrainingConfig, model: torch.nn.Module):
        if optimizer_type == cls.SGD:
            return torch.optim.SGD(
                model.parameters(),
                lr=config.hyperparameters.learning_rate,
                momentum=config.hyperparameters.additional_params["momentum"],
            )
        elif optimizer_type == cls.ADAM:
            return torch.optim.Adam(
                model.parameters(),
                lr=config.hyperparameters.learning_rate,
                weight_decay=config.hyperparameters.additional_params.get("weight_decay", 0.0),
            )
        else:
            raise ValueError(f"Invalid optimizer: {config.hyperparameters.optimizer}")
    
    @classmethod
    def is_optimizer(cls, optimizer_type: str):
        return optimizer_type in [o.value for o in cls]

class LRScheduler(StrEnum):
    StepLR = auto()

    # class method to build scheduler with type, optimizer, and config
    @classmethod
    def build(cls, scheduler_type: str, optimizer: torch.optim.Optimizer, config: TrainingConfig):
        if scheduler_type is None:
            return None
        elif scheduler_type == cls.StepLR:
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.hyperparameters.additional_params["step_size"],
                gamma=config.hyperparameters.additional_params["gamma"],
            )
        else:
            raise ValueError(f"Invalid scheduler: {config.hyperparameters.scheduler}")
    
    @classmethod
    def is_scheduler(cls, scheduler_type: str):
        return scheduler_type in [s.value for s in cls]

