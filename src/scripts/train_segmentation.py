import torch
from torchvision import transforms
import argparse
from dotenv import load_dotenv
import os
from enum import StrEnum, auto
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC, BinaryAccuracy, Dice, BinaryPrecision, BinaryRecall
)

torch.set_printoptions(precision=3, edgeitems=40, linewidth=120, sci_mode=False)

from src.utils.transforms import DualInputCompose, DualInputResize, DualInputTransform
from src.utils.config import TrainingConfig, Hyperparameters
from src.metrics import BinaryIoU
from src.utils.logging import WandbLogger, LocalLogger
from src.data.datasets import prepare_datasets, DatasetType
from src.trainer import k_fold_cross_validation, train_model

"""
TODO: Class imbalance handling for LGG dataset
TODO: Add a scheduler to adjust learning rate
TODO: Look into checkpointing for training (e.g. save model every n epochs instead of just the best model)
TODO: Look into AMP (Automatic Mixed Precision) for faster training (useful for large models) (use torch.cuda.amp.autocast() and torch.cuda.amp.GradScaler()) (requires NVIDIA GPU with Tensor Cores)
TODO: Include hyperparameter info in saved model file name
TODO: Differentiate between test and validation set metrics (e.g. val_loss vs test_loss)
"""

class SegmentationArchitecture(StrEnum):
    UNET = auto()

# LOGIN TO W&B
load_dotenv()


"""
Key Notes for Later Improvements / Implementations Details:
- NOTE: do we need a scheduler to adjust learning rate?
- NOTE: UNet also introduces pre-computed weights for the loss function to handle class imbalance
"""


def get_train_config():
    """
    Get the training configuration
    """
    # parse args
    parser = argparse.ArgumentParser(
        description="Train a model with specified parameters."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Input batch size for training (default: 1)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=10, help="Number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds for cross validation (default: 5)",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Flag to enable logging to W&B (default: False)",
    )

    # only allow options `box` and `llr` for now
    parser.add_argument(
        '--dataset',
        type=DatasetType,
        default=DatasetType.BOX,
        help='Dataset to use for training (default: box) (options: box, lgg)',
    )

    parser.add_argument(
        "--architecture",
        type=SegmentationArchitecture,
        default=SegmentationArchitecture.UNET,
        help="Segmentation architecture to use for training (default: unet) (options: unet)"
    )

    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Flag to save the best model (default: False). If k-fold cross validation is used, only the best model will be saved per fold",
    )

    args = parser.parse_args()

    # TODO: accept more model architectures as input
    assert args.architecture in ["unet"], f"Invalid architecture: {args.architecture}"

    # TODO: add metrics to the configuration
    # create a training configuration
    hyperparams = Hyperparameters(
        optimizer="SGD",
        loss_fn="BCEWithLogitsLoss",
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
        additional_params={"momentum": 0.99},
    )

    return TrainingConfig(
        architecture=args.architecture,
        dataset=args.dataset,
        n_folds=args.n_folds,
        use_wandb=args.use_wandb,
        hyperparameters=hyperparams,
    )

def main():
    # Get the training configuration
    training_config = get_train_config()

    if training_config.use_wandb:
        logger = WandbLogger(training_config, os.getenv("WANDB_API_KEY"), project_name=os.getenv("WANDB_PROJECT"), tags=["segmentation", training_config.architecture, training_config.dataset])
    else:
        logger = LocalLogger(training_config)

    base_transforms = DualInputCompose(
        [DualInputResize((320, 320)), DualInputTransform(transforms.ToTensor())]
    )

    train_dataset, test_dataset = prepare_datasets(training_config, base_transforms)

    # Define Metrics to track (e.g. accuracy, precision, recall, etc.)
    # NOTE: Metrics are defined in the torchmetrics library however, custom metrics can be created if needed
    metrics = MetricCollection(
        [
            BinaryAUROC().cpu(),
            BinaryIoU(multidim_average='samplewise', threshold=0.5).cpu(),
            BinaryAccuracy().cpu(),
            Dice(average='samples', threshold=0.5).cpu(),
            BinaryPrecision(multidim_average='samplewise', threshold=0.5).cpu(),
            BinaryRecall(multidim_average='samplewise', threshold=0.5).cpu()
        ]
    )
    
    # Train the model
    if training_config.n_folds > 1:
        k_fold_cross_validation(training_config, train_dataset, metrics, logger)
    else:
        train_model(training_config, train_dataset, test_dataset, metrics, logger)
        

if __name__ == "__main__":
    main()
