import torch
import numpy as np
from torchvision import transforms
import argparse
from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    Dice,
    BinaryPrecision,
    BinaryRecall,
)
from datetime import datetime

torch.set_printoptions(precision=3, edgeitems=40, linewidth=120, sci_mode=False)

from src.utils.transforms import DualInputCompose, DualInputResize, DualInputTransform
from src.utils.config import TrainingConfig, Hyperparameters
from src.metrics import BinaryIoU
from src.utils.logging import WandbLogger, LocalLogger
from src.data.datasets import prepare_datasets, DatasetType
from src.trainer import k_fold_cross_validation, train_model
from src.models.model_utils import LRScheduler, Optimizer, SegmentationArchitecture

"""
TODO: Class imbalance handling for LGG dataset
TODO: Add a scheduler to adjust learning rate
TODO: Look into checkpointing for training (e.g. save model every n epochs instead of just the best model)
TODO: Look into AMP (Automatic Mixed Precision) for faster training (useful for large models) (use torch.cuda.amp.autocast() and torch.cuda.amp.GradScaler()) (requires NVIDIA GPU with Tensor Cores)
TODO: Include hyperparameter info in saved model file name
TODO: Differentiate between test and validation set metrics (e.g. val_loss vs test_loss)
"""

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
        "--n_epochs",
        type=int,
        default=10,
        help="Number of epochs to train (default: 10)",
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
        "--dataset",
        type=DatasetType,
        default=DatasetType.BOX,
        help='Dataset to use for training (default: box) (options: box, lgg, lgg_norm)',
    )

    parser.add_argument(
        "--architecture",
        type=SegmentationArchitecture,
        default=SegmentationArchitecture.UNET,
        help="Segmentation architecture to use for training (default: unet) (options: unet)",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=LRScheduler,
        help="Learning rate scheduler to use for training (default: None) (options: None, StepLR)",
    )

    parser.add_argument(
        "--n_checkpoints",
        type=int,
        default=1,
        help="Number of checkpoints to save during training (default: 1)",
    )

    args = parser.parse_args()

    # TODO: accept more model architectures as input
    assert args.architecture in ["unet"], f"Invalid architecture: {args.architecture}"

    additional_params = {"momentum": 0.99}

    if args.lr_scheduler is LRScheduler.StepLR:
        additional_params["step_size"] = 20
        additional_params["gamma"] = 0.1

    # TODO: add metrics to the configuration
    # create a training configuration
    hyperparams = Hyperparameters(
        optimizer=Optimizer.SGD,
        scheduler=args.lr_scheduler,
        loss_fn="BCEWithLogitsLoss",
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
        additional_params=additional_params,
    )

    return TrainingConfig(
        architecture=args.architecture,
        dataset=args.dataset,
        n_folds=args.n_folds,
        n_checkpoints=args.n_checkpoints,
        use_wandb=args.use_wandb,
        hyperparameters=hyperparams,
    )


def main():
    # Get the training configuration
    training_config = get_train_config()

    if training_config.use_wandb:
        logger = WandbLogger(
            training_config,
            os.getenv("WANDB_API_KEY"),
            project_name=os.getenv("WANDB_PROJECT"),
            tags=[
                "segmentation",
                training_config.architecture,
                training_config.dataset,
            ],
        )
    else:
        logger = LocalLogger(
            training_config,
            run_group=f"Train Segmentation {training_config.architecture} {training_config.dataset} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            checkpointing=True,
        )

    base_transforms = DualInputCompose(
        [DualInputResize((320, 320)), DualInputTransform(transforms.ToTensor())]
    )

    train_dataset, test_dataset = prepare_datasets(training_config, base_transforms)

    # Define Metrics to track (e.g. accuracy, precision, recall, etc.)
    # NOTE: Metrics are defined in the torchmetrics library however, custom metrics can be created if needed
    metrics = MetricCollection(
        [
            BinaryAUROC().cpu(),
            BinaryIoU(multidim_average="samplewise", threshold=0.5).cpu(),
            BinaryAccuracy().cpu(),
            Dice(average="samples", threshold=0.5).cpu(),
            BinaryPrecision(multidim_average="samplewise", threshold=0.5).cpu(),
            BinaryRecall(multidim_average="samplewise", threshold=0.5).cpu(),
        ]
    )

    # Train the model
    if training_config.n_folds > 1:
        k_fold_cross_validation(training_config, train_dataset, metrics, logger)
    else:
        val_size = 0.2
        train_dataset_size = len(train_dataset)
        train_idx, val_idx = train_test_split(
            np.arange(train_dataset_size),
            test_size=val_size,
            random_state=training_config.random_state,
        )
        sub_train_dataset = Subset(train_dataset, train_idx)
        val_dataset = Subset(train_dataset, val_idx)
        train_model(training_config, sub_train_dataset, val_dataset, metrics, logger)


if __name__ == "__main__":
    main()
