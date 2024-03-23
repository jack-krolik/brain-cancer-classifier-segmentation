from src.models.classification.efficient_net import MODEL_MODES, EfficientNet
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from torchinfo import summary
import torchmetrics
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryJaccardIndex,
)
from sklearn.model_selection import KFold
from tqdm import trange, tqdm
import argparse
import pathlib
from dataclasses import dataclass, field

torch.set_printoptions(precision=3, edgeitems=40, linewidth=120, sci_mode=False)

from src.models.segmentation.unet import UNet
from src.data.classification import TumorBinaryClassificationDataset
from src.utils.visualize import show_images_with_masks
from src.utils.transforms import (
    DualInputCompose,
    DualInputResize,
    DualInputTransform,
    SingleInputCompose,
    SingleInputResize,
    SingleInputTransform,
)


"""
Key Notes for Later Improvements / Implementations Details:
- NOTE: UNet paper uses SGD with momentum = 0.99 (this may be a good starting point for hyperparameter tuning)
- NOTE: do we need a scheduler to adjust learning rate?
- NOTE: UNet also introduces pre-computed weights for the loss function to handle class imbalance
- NOTE: BCEWithLogitsLoss is used for binary segmentation tasks (like this one) and is a combination of Sigmoid and BCELoss
- NOTE: Main benefit of BCEWithLogitsLoss is that it is numerically stable because it uses the log-sum-exp trick
"""

DATASET_BASE_DIR = pathlib.Path(__file__).parent.parent.parent / "datasets"


def get_device():
    """
    Get the device to use for training (cuda if available, then mps, then cpu)
    """
    if torch.cuda.is_available():
        print("Running on CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Running on MPS")
        return torch.device("mps")
    else:
        print("Running on CPU")
        return torch.device("cpu")


@dataclass
class TrainingConfig:
    batch_size: int = 1
    learning_rate: float = 0.01
    num_epochs: int = 10
    device: torch.device = field(default_factory=lambda: get_device())
    dataset_root_dir: str = DATASET_BASE_DIR
    num_folds: int = 5
    random_state: int = 42
    optimizer: str = "SGD"
    momentum: float = 0.99


def get_config():
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
    args = parser.parse_args()

    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
    )
    # validate the configuration
    assert config.batch_size > 0, "Batch size must be greater than 0"
    assert 1 > config.learning_rate > 0, "Learning rate must be greater than 0"
    assert config.num_epochs > 0, "Number of epochs must be greater than 0"
    assert pathlib.Path(
        config.dataset_root_dir
    ).exists(), "Dataset root directory does not exist"

    return config


def train(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    config: TrainingConfig,
):
    """
    Train the model on the training set

    Args:
    - model (torch.nn.Module): the model to train
    - train_dataloader (DataLoader): the training set dataloader
    - optimizer (torch.optim.Optimizer): the optimizer to use
    - loss_fn (torch.nn.Module): the loss function to use
    - config (dict): a dictionary containing the training configuration
    """
    device, num_epochs = config.device, config.num_epochs
    model.train()
    for epoch in range(num_epochs):
        cumalative_loss = 0
        with tqdm(
            total=len(train_dataloader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
            unit="batch",
        ) as pbar:
            for x_batch, y_batch in train_dataloader:

                x_batch, y_batch = x_batch.to(device), y_batch.to(device).float()
                model.zero_grad()
                output = model(x_batch).float()

                output = output.view(-1)

                loss = loss_fn(output, y_batch)
                loss.backward()
                optimizer.step()
                cumalative_loss += loss.item()

                # NOTE: we can add additional metrics here (e.g. accuracy, precision, recall, etc.)

                pbar.set_postfix({"Loss": loss.item()})
                pbar.update()

        print(
            f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {cumalative_loss / len(train_dataloader)}"
        )
        # TODO: Add a scheduler to adjust learning rate
        # May also want to evaluate the model on the validation set after each epoch (or every few epochs)


def evaluate(
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    config: TrainingConfig,
    metrics: torchmetrics.MetricCollection,
):
    """
    Evaluate the model on the validation set

    Args:
    - model (torch.nn.Module): the model to evaluate
    - dataloader (DataLoader): the validation set dataloader
    - loss_fn (torch.nn.Module): the loss function to use
    """
    device = config.device

    # Reset the metrics
    metrics.reset()

    model.eval()
    with torch.no_grad():
        cumalative_loss = 0
        with tqdm(total=len(val_dataloader), desc="Validation", unit="batch") as pbar:
            for x_batch, y_batch in val_dataloader:

                x_batch, y_batch = x_batch.to(device), y_batch.to(device).float()
                model.zero_grad()
                output = model(x_batch).float()

                output = output.view(-1)
                loss = loss_fn(output, y_batch)
                cumalative_loss += loss.item()

                preds = output.detach().sigmoid().round()
                computed_metrics = metrics(preds, y_batch)
                pbar.set_postfix({"Loss": loss.item()})
                pbar.update()

        total_metrics = (
            metrics.compute()
        )  # Compute the metrics over the entire validation set
        print(f"Validation Metrics: {total_metrics}")
        print(f"Validation Loss: {cumalative_loss / len(val_dataloader)}")


def main():
    # Get the training configuration
    config = get_config()
    device = config.device

    # Define the augmentation pipeline for the dataset
    # TODO: Here is where augmentation should be added to the dataset
    base_transforms = SingleInputCompose(
        [SingleInputResize((224, 224)), SingleInputTransform(transforms.ToTensor())]
    )

    # Create Segmentation Dataset instance
    train_dataset = TumorBinaryClassificationDataset(
        root_dir=config.dataset_root_dir, split="train", transform=base_transforms
    )
    test_dataset = TumorBinaryClassificationDataset(
        root_dir=config.dataset_root_dir, split="test", transform=base_transforms
    )

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    # Create a KFold instance
    kfold = KFold(
        n_splits=config.num_folds, shuffle=True, random_state=config.random_state
    )

    # Define Metrics to track (e.g. accuracy, precision, recall, etc.)
    # NOTE: Metrics are defined in the torchmetrics library however, custom metrics can be created if needed
    metrics = torchmetrics.MetricCollection(
        [
            BinaryAUROC().to(device),
            BinaryJaccardIndex().to(device),
            BinaryAccuracy().to(device),
            BinaryF1Score().to(device),
            BinaryPrecision().to(device),
            BinaryRecall().to(device),
        ]
    )

    # Train the model using k-fold cross validation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
        print(f"Fold {fold}")
        print("---------------------------")

        # randomly sample train and validation ids from the dataset based on the fold
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)

        train_dataloader = DataLoader(
            train_dataset, batch_size=config.batch_size, sampler=train_sampler
        )
        val_dataloader = DataLoader(
            train_dataset, batch_size=config.batch_size, sampler=val_sampler
        )

        # Create a UNet model to train on this fold
        model = EfficientNet("b0")
        # summary(model, input_size=(config.batch_size, 3, 320, 320)) # TODO - programatically get input size

        # Move the model to the device
        model.to(device)

        if config.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=config.learning_rate, momentum=config.momentum
            )
        else:
            # NOTE: add more optimizers as needed
            raise ValueError(f"Invalid optimizer: {config.optimizer}")

        loss_fn = torch.nn.BCEWithLogitsLoss(
            reduction="mean"
        )  # Binary Cross Entropy with Logits Loss

        # Train the model
        train(model, train_dataloader, optimizer, loss_fn, config)

        # Evaluate the model
        evaluate(model, val_dataloader, loss_fn, config, metrics)


if __name__ == "__main__":
    main()
