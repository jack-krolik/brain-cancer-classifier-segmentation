import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from sklearn.model_selection import KFold
from tqdm import tqdm
import argparse
from argparse import Namespace
import pathlib
from dotenv import load_dotenv
import os
from enum import StrEnum, auto
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC, BinaryAccuracy, Dice, BinaryPrecision, BinaryRecall
)

torch.set_printoptions(precision=3, edgeitems=40, linewidth=120, sci_mode=False)

from src.models.segmentation.unet import UNet
from src.data.segmentation import BoxSegmentationDataset, LGGSegmentationDataset
from src.utils.transforms import DualInputCompose, DualInputResize, DualInputTransform
from src.utils.config import TrainingConfig, Hyperparameters
from src.utils.wandb import create_wandb_config, verify_wandb_config, wandb_init
from src.metrics import BinaryIoU
from src.utils.logging import WandbLogger, LocalLogger, LoggerMixin

# TODO: Move this to a separate file

"""
TODO: Class imbalance handling for LGG dataset
TODO: Include information about total training time of folds (maybe make pbar more dynamic)
TODO: Add a scheduler to adjust learning rate
TODO: Look into checkpointing for training (e.g. save model every n epochs instead of just the best model)
TODO: Look into AMP (Automatic Mixed Precision) for faster training (useful for large models) (use torch.cuda.amp.autocast() and torch.cuda.amp.GradScaler()) (requires NVIDIA GPU with Tensor Cores)
TODO: Include hyperparameter info in saved model file name
TODO: Allow single fold training (No KFold) this would be for training the final model
"""

class DatasetType(StrEnum):
    BOX = auto()
    LGG = auto()

class SegmentationArchitecture(StrEnum):
    UNET = auto()

# LOGIN TO W&B
load_dotenv()


SAVE_MODEL_DIR = pathlib.Path(__file__).parent.parent.parent / "model_registry"

if not SAVE_MODEL_DIR.exists():
    SAVE_MODEL_DIR.mkdir()

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


def main_train_loop(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    config: dict,
    metrics: MetricCollection,
    device: torch.device,
    logger=LoggerMixin,
) -> pathlib.Path:
    """
    Main training loop for the model

    Args:
    - model (torch.nn.Module): the model to train
    - train_dataloader (DataLoader): the training set dataloader
    - val_dataloader (DataLoader): the validation set dataloader
    - optimizer (torch.optim.Optimizer): the optimizer algorithm (e.g. SGD, Adam, etc.)
    - loss_fn (torch.nn.Module): the loss function being optimized
    - metrics (MetricCollection): the metrics to track
    - logger (Logger): the logger to use for tracking metrics

    Returns:
    - pathlib.Path: the path to the best model
    """
    num_epochs = config.n_epochs
    total_steps = len(train_dataloader) + len(val_dataloader)

    model.to(device)

    best_eval_loss = float("inf")
    best_model_path = None

    for epoch in range(num_epochs):
        with tqdm(
            total=total_steps, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"
        ) as pbar:
            train_loss = train(
                model, train_dataloader, optimizer, loss_fn, config, device, pbar
            )

            val_metrics = evaluate(
                model, val_dataloader, loss_fn, metrics, device, pbar
            )

            metrics_bundled = {
                f"val_{metric_name}": metric_value
                for metric_name, metric_value in val_metrics.items()
            }
            metrics_bundled["train_loss"] = (
                train_loss  # add the training loss to the metrics
            )

            logger.log_metrics(metrics_bundled)

            val_loss = metrics_bundled["val_loss"]

            if val_loss < best_eval_loss:

                if best_model_path is not None:  # remove the previous best model
                    os.remove(best_model_path)

                # save the best model locally
                best_model_path = (
                    SAVE_MODEL_DIR
                    / f"best_{config.architecture}_{config.dataset}_fold_{config.fold}_model.h5"
                )
                torch.save(model.state_dict(), best_model_path)
                best_eval_loss = val_loss

                print(f"New best model saved at {best_model_path}")
    return best_model_path


def train(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    config: dict,
    device: torch.device,
    pbar: tqdm,
):
    """
    Train the model on the training set

    Args:
    - model (torch.nn.Module): the model to train
    - train_dataloader (DataLoader): the training set dataloader
    - optimizer (torch.optim.Optimizer): the optimizer algorithm (e.g. SGD, Adam, etc.)
    - loss_fn (torch.nn.Module): the loss function being optimized
    - config (dict): the wandb configuration for the training
    - device (torch.device): the device to use for training
    - pbar (tqdm): the progress bar to update
    """
    num_epochs = config.n_epochs
    model.train()
    cumalative_loss = 0
    optimizer.zero_grad()
    for i, (imgs, masks) in enumerate(train_dataloader):
        imgs, masks = imgs.to(device), masks.to(device)
        output = model(imgs)
        loss = loss_fn(output, masks)
        loss.backward()
        if (i + 1) % config.accumulation_steps == 0: # Gradient accumulation
            optimizer.step()
            optimizer.zero_grad()

        cumalative_loss += loss.item()

        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Phase": "Train"})
        pbar.update()

    # TODO: Add a scheduler to adjust learning rate

    train_loss = cumalative_loss / len(train_dataloader)

    return train_loss


def evaluate(
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    metrics: MetricCollection,
    device: torch.device,
    pbar: tqdm,
):
    """
    Evaluate the model on the validation set

    Args:
    - model (torch.nn.Module): the model to evaluate
    - dataloader (DataLoader): the validation set dataloader
    - loss_fn (torch.nn.Module): the loss function to use
    - config (TrainingConfig): the training configuration
    - metrics (MetricCollection): the metrics to track
    - device (torch.device): the device to use for evaluation
    - pbar (tqdm): the progress bar to update
    """
    model.eval()
    # Reset the metrics
    metrics.reset()
    with torch.inference_mode():
        cumalative_loss = 0
        for imgs, masks in val_dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            output = model(imgs)
            loss = loss_fn(output, masks)
            cumalative_loss += loss.item()

            preds = output.detach()

            metrics(preds.cpu(), masks.type(torch.int32).cpu())
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Phase": "Validation"})
            pbar.update()

    total_metrics = (
        metrics.compute()
    )  # Compute the metrics over the entire validation set

    # average metrics over samples
    for metric_name, metric_value in total_metrics.items():
        total_metrics[metric_name] = torch.mean(metric_value)

    return {"loss": cumalative_loss / len(val_dataloader), **total_metrics}

def prepare_datasets(config: TrainingConfig):
    """
    Prepare the datasets for training

    Args:
    - config (TrainingConfig): the training configuration

    Returns:
    - Tuple[DataLoader, DataLoader]: the training and testing dataloaders
    """
    # Define the augmentation pipeline for the dataset

    # NOTE: ALL AUGMENTATIONS SHOULD BE ADDED HERE
    base_transforms = DualInputCompose(
        [DualInputResize((320, 320)), DualInputTransform(transforms.ToTensor())]
    )

    if config.dataset == DatasetType.BOX:
        # Create Segmentation Dataset instance
        train_dataset = BoxSegmentationDataset(
            root_dir=config.dataset_root_dir,
            split="train",
            transform=base_transforms,
        )
        test_dataset = BoxSegmentationDataset(
            root_dir=config.dataset_root_dir,
            split="test",
            transform=base_transforms,
        )
    elif config.dataset == DatasetType.LGG:
        # Create Segmentation Dataset instance
        train_dataset = LGGSegmentationDataset(
            root_dir=config.dataset_root_dir,
            split="train",
            transform=base_transforms,
        )
        test_dataset = LGGSegmentationDataset(
            root_dir=config.dataset_root_dir,
            split="test",
            transform=base_transforms,
        )
    else:
        raise ValueError(f"Invalid dataset: {config.dataset}")

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")
    
    return train_dataset, test_dataset

def build_model_from_config(config: TrainingConfig):
    """
    Build model, optimizer, and loss function from the training configuration

    Args:
    - config (TrainingConfig): the training configuration

    Returns:
    - torch.nn.Module: the model to train
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

def main():
    # Get the training configuration
    training_config = get_train_config()
    device = training_config.device

    if training_config.use_wandb:
        logger = WandbLogger(training_config, os.getenv("WANDB_API_KEY"))
    else:
        logger = LocalLogger(training_config)

    train_dataset, test_dataset = prepare_datasets(training_config)

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

    # Create a KFold instance
    kfold = KFold(
        n_splits=training_config.n_folds,
        shuffle=True,
        random_state=training_config.random_state,
    )

    # Train the model using k-fold cross validation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
        print(f"Fold {fold}")
        print("---------------------------")

        try:
            logger.init(project=os.getenv("WANDB_PROJECT"), tags=["segmentation", training_config.architecture, training_config.dataset, f"fold_{fold}"])

            config = Namespace(**training_config.flatten())

            # TODO: determine how to bundle the dataset into a make method
            # randomly sample train and validation ids from the dataset based on the fold
            train_sampler = SubsetRandomSampler(train_ids)
            val_sampler = SubsetRandomSampler(val_ids)

            train_dataloader = DataLoader(
                train_dataset, batch_size=config.batch_size, sampler=train_sampler
            )
            val_dataloader = DataLoader(
                train_dataset, batch_size=config.batch_size, sampler=val_sampler
            )

            model, optimizer, loss_fn = build_model_from_config(training_config)
            
            # Train the model
            path_to_best_model = main_train_loop(
                model,
                train_dataloader,
                val_dataloader,
                optimizer,
                loss_fn,
                config,
                metrics,
                device,
                logger=logger,
            )

            # TODO: Currently, not model is saved for k-fold cross validation
            # if training_config.use_wandb and training_config.save_model:
            #     logger.save_model(str(path_to_best_model))
        finally:
            logger.finish()
        

if __name__ == "__main__":
    main()
