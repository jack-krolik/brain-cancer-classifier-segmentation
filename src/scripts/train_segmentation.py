import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from torchinfo import summary
from sklearn.model_selection import KFold
from tqdm import trange, tqdm
import argparse
from argparse import Namespace
import pathlib
from dotenv import load_dotenv
import wandb
import os
from enum import StrEnum, auto

torch.set_printoptions(precision=3, edgeitems=40, linewidth=120, sci_mode=False)

from src.models.segmentation.unet import UNet
from src.data.segmentation import BoxSegmentationDataset, LGGSegmentationDataset
from src.utils.transforms import DualInputCompose, DualInputResize, DualInputTransform
from src.utils.config import TrainingConfig, Hyperparameters
from src.utils.wandb import create_wandb_config, verify_wandb_config, wandb_init
import src.metrics as Metrics

# TODO: Move this to a separate file

"""
TODO: Write own metrics instead of using torchmetrics
TODO: Class imbalance handling for LGG dataset
TODO: Include information about total training time of folds (maybe make pbar more dynamic)
TODO: Add a scheduler to adjust learning rate
TODO: Look into checkpointing for training (e.g. save model every n epochs instead of just the best model)
TODO: Look into Gradient Accumulation (e.g. accumulate gradients over multiple batches before updating the model) (useful for large batch sizes and not crashing GPU)
TODO: Look into AMP (Automatic Mixed Precision) for faster training (useful for large models) (use torch.cuda.amp.autocast() and torch.cuda.amp.GradScaler()) (requires NVIDIA GPU with Tensor Cores)
TODO: Include hyperparameter info in saved model file name
TODO: Make general printing of metrics more dynamic and cleaner
"""



class DatasetType(StrEnum):
    BOX = auto()
    LGG = auto()

# LOGIN TO W&B
load_dotenv()


SAVE_MODEL_DIR = pathlib.Path(__file__).parent.parent.parent / "model_registry"

if not SAVE_MODEL_DIR.exists():
    SAVE_MODEL_DIR.mkdir()

"""
Key Notes for Later Improvements / Implementations Details:
- NOTE: UNet paper uses SGD with momentum = 0.99 (this may be a good starting point for hyperparameter tuning)
- NOTE: do we need a scheduler to adjust learning rate?
- NOTE: UNet also introduces pre-computed weights for the loss function to handle class imbalance
- NOTE: BCEWithLogitsLoss is used for binary segmentation tasks (like this one) and is a combination of Sigmoid and BCELoss
- NOTE: Main benefit of BCEWithLogitsLoss is that it is numerically stable because it uses the log-sum-exp trick
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
        "--epochs", type=int, default=10, help="Number of epochs to train (default: 10)"
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

    parser.add_argument("--architecture", type=str, default="unet")
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
        n_epochs=args.epochs,
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
    metrics: Metrics.MetricsPipeline,
    device: torch.device,
    logger=print,
) -> pathlib.Path:
    """
    Main training loop for the model

    Args:
    - model (torch.nn.Module): the model to train
    - train_dataloader (DataLoader): the training set dataloader
    - val_dataloader (DataLoader): the validation set dataloader
    - optimizer (torch.optim.Optimizer): the optimizer algorithm (e.g. SGD, Adam, etc.)
    - loss_fn (torch.nn.Module): the loss function being optimized
    - metrics (Metrics.MetricsPipeline): the metrics to validate the model performance
    - logger (Callable): the logger to use for logging (e.g. wandb.log, print, etc.)

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

            logger(metrics_bundled)

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
    for imgs, masks in train_dataloader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = loss_fn(output, masks)
        loss.backward()
        optimizer.step()
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
    metrics: Metrics.MetricsPipeline,
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
    - metrics (Metrics.MetricsPipeline): the metrics to track
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

            preds = output.detach().sigmoid().round()

            computed_metrics = metrics.update(preds, masks)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Phase": "Validation"})
            pbar.update()

    total_metrics = (
        metrics.compute_final()
    )  # Compute the metrics over the entire validation set
    return {"loss": cumalative_loss / len(val_dataloader), **total_metrics}


def main():
    # Get the training configuration
    training_config = get_train_config()
    device = training_config.device

    # Login to W&B
    if training_config.use_wandb:
        wandb.login(key=os.getenv("WANDB_API_KEY"), verify=True)

    # Define the augmentation pipeline for the dataset
    # TODO: Here is where augmentation should be added to the dataset
    base_transforms = DualInputCompose(
        [DualInputResize((320, 320)), DualInputTransform(transforms.ToTensor())]
    )

    if training_config.dataset == DatasetType.BOX:
        # Create Segmentation Dataset instance
        train_dataset = BoxSegmentationDataset(
            root_dir=training_config.dataset_root_dir,
            split="train",
            transform=base_transforms,
        )
        test_dataset = BoxSegmentationDataset(
            root_dir=training_config.dataset_root_dir,
            split="test",
            transform=base_transforms,
        )
    elif training_config.dataset == DatasetType.LGG:
        # Create Segmentation Dataset instance
        train_dataset = LGGSegmentationDataset(
            root_dir=training_config.dataset_root_dir,
            split="train",
            transform=base_transforms,
        )
        test_dataset = LGGSegmentationDataset(
            root_dir=training_config.dataset_root_dir,
            split="test",
            transform=base_transforms,
        )
    else:
        raise ValueError(f"Invalid dataset: {training_config.dataset}")

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    # Create a KFold instance
    kfold = KFold(
        n_splits=training_config.n_folds,
        shuffle=True,
        random_state=training_config.random_state,
    )

    # Define Metrics to track (e.g. accuracy, precision, recall, etc.)
    # NOTE: Metrics are defined in the torchmetrics library however, custom metrics can be created if needed
    metrics = Metrics.MetricsPipeline([
        Metrics.ConfusionMatrixMetric(num_classes=2), # NOTE: num_classes should be 2 for binary segmentation tasks CHANGE IF MULTICLASS
        Metrics.AccuracyMetric(),
        Metrics.PrecisionMetric(),
        Metrics.RecallMetric(),
        Metrics.F1ScoreMetric(),
        Metrics.IOUMetric(),
        # Metrics.AUCMetric(), # NOTE: AUCMetric is not implemented yet
    ])

    # Train the model using k-fold cross validation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
        print(f"Fold {fold}")
        print("---------------------------")

        wandb_config = create_wandb_config(training_config, {"fold": fold})

        with wandb_init(
            wandb_config,
            enabled_wandb=training_config.use_wandb,
            project=os.getenv("WANDB_PROJECT"),
            tags=["segmentation"],
        ):
            if training_config.use_wandb:
                config = wandb.config
                assert verify_wandb_config(
                    config, training_config
                ), "W&B config does not match training config"
            else:
                config = Namespace(**wandb_config)

            # TODO: determine how to bundle the dataset into a make method
            # randomly sample train and validation ids from the dataset based on the fold
            train_sampler = SubsetRandomSampler(train_ids[:100]) # TODO: remove the 100 limit
            val_sampler = SubsetRandomSampler(val_ids)

            train_dataloader = DataLoader(
                train_dataset, batch_size=config.batch_size, sampler=train_sampler
            )
            val_dataloader = DataLoader(
                train_dataset, batch_size=config.batch_size, sampler=val_sampler
            )

            if config.architecture == "unet":
                model = UNet()
            else:
                raise ValueError(f"Invalid architecture: {config.architecture}")

            if config.optimizer == "SGD":
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=config.learning_rate,
                    momentum=config.momentum,
                )
            else:
                # NOTE: add more optimizers as needed
                raise ValueError(f"Invalid optimizer: {config.optimizer}")

            if config.loss_fn == "BCEWithLogitsLoss":
                loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
            
            logger = wandb.log if not training_config.use_wandb else print

            # Train the model
            path_to_best_model =  main_train_loop(
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

            if not training_config.use_wandb:
                wandb.save(str(path_to_best_model))


if __name__ == "__main__":
    main()
