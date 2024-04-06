import torch
from typing import Tuple

from src.utils.config import TrainingConfig
from src.models.segmentation.unet import UNet


def build_model_from_config(config: TrainingConfig) -> Tuple[torch.nn.Module, torch.optim.Optimizer, torch.nn.Module]:
    """
    Build model, optimizer, and loss function from the training configuration

    Args:
    - config (TrainingConfig): the training configuration

    Returns:
    - torch.nn.Module: the model to train,
    """

    if config.architecture == "unet":
        model = UNet()
    else:
        raise ValueError(f"Invalid architecture: {config.architecture}")
    
    if config.hyperparameters.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.hyperparameters.learning_rate,
            momentum=config.hyperparameters.additional_params["momentum"],
        )
    else:
        raise ValueError(f"Invalid optimizer: {config.hyperparameters.optimizer}")

    if config.hyperparameters.loss_fn == "BCEWithLogitsLoss":
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
    else:
        raise ValueError(f"Invalid loss function: {config.hyperparameters.loss_fn}")

    # NOTE (TODO): Add more optimizers and loss functions as needed
    # NOTE (TODO): Add more model architectures as needed
    # NOTE: add scheduler to adjust learning rate
    
    return model, optimizer, loss_fn