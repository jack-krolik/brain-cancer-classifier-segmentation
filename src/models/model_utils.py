import torch
from typing import Tuple, Optional
from enum import auto, StrEnum

from src.utils.config import TrainingConfig, Optimizer, LRScheduler
from src.models.segmentation.unet import UNet

class SegmentationArchitecture(StrEnum):
    UNET = auto()

def build_model_from_config(config: TrainingConfig) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler], torch.nn.Module]:
    """
    Build model, optimizer, lr scheduler, and loss function from the training configuration

    Args:
    - config (TrainingConfig): the training configuration

    Returns:
    - torch.nn.Module: the model to train,
    """

    if config.architecture == "unet":
        model = UNet()
    else:
        raise ValueError(f"Invalid architecture: {config.architecture}")
    
    optimizer = Optimizer.build(config.hyperparameters.optimizer, config, model)
    lr_scheduler = LRScheduler.build(config.hyperparameters.scheduler, optimizer, config)

    if config.hyperparameters.loss_fn == "BCEWithLogitsLoss":
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
    else:
        raise ValueError(f"Invalid loss function: {config.hyperparameters.loss_fn}")

    # NOTE (TODO): Add more optimizers and loss functions as needed
    # NOTE (TODO): Add more model architectures as needed
    # NOTE: add scheduler to adjust learning rate
    
    return model, optimizer, lr_scheduler, loss_fn