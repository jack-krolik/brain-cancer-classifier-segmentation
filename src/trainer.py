import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchmetrics import MetricCollection
import pathlib
from tqdm import tqdm
from argparse import Namespace
from sklearn.model_selection import KFold

from src.models.model_utils import build_model_from_config
from src.utils.config import TrainingConfig
from src.utils.logging import LoggerMixin


def run_training_and_evaluation_cycle(
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

    metrics_bundled = {}

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


            # NOTE: checkpoints not implemented
            

    return metrics_bundled # last computed metrics


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

def k_fold_cross_validation(config: TrainingConfig, train_dataset: Dataset, metrics: MetricCollection, logger: LoggerMixin):
    """
    Perform k-fold cross validation on the given training dataset

    Args:
    - config (TrainingConfig): the training configuration
    - train_dataset (Dataset): the training dataset
    - metrics (MetricCollection): the metrics to track
    - logger (LoggerMixin): logging object to track metrics, models, and visuals
    """
    # Create a KFold instance
    kfold = KFold(
        n_splits=config.n_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    avg_metrics = {}

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

        train_metrics = train_model(config, train_dataloader, val_dataloader, metrics, logger)

        for metric_name, metric_value in train_metrics.items():
            if metric_name in avg_metrics:
                avg_metrics[metric_name] += (metric_value - avg_metrics[metric_name]) / (fold + 1) # update the average
            else:
                avg_metrics[metric_name] = metric_value
    
    return avg_metrics # average metrics over all folds

def train_model(config: TrainingConfig, train_dataset: Dataset, test_dataset: Dataset, metrics: MetricCollection, logger: LoggerMixin):
    """
    Train a segmentation model using the given configuration, datasets, metrics, and logger

    Args:
    - config (TrainingConfig): the training configuration
    - train_dataset (Dataset): the training dataset
    - test_dataset (Dataset): the testing dataset (or validation dataset if using k-fold cross validation)
    - metrics (MetricCollection): the metrics to track
    - logger (LoggerMixin): logging object to track metrics, models, and visuals
    """
    try:
        logger.init()
        config = Namespace(**config.flatten())
        device = config.device

        model, optimizer, loss_fn = build_model_from_config(config)
        
        # Train the model
        computed_metrics = run_training_and_evaluation_cycle(
            model,
            train_dataset,
            test_dataset,
            optimizer,
            loss_fn,
            config,
            metrics,
            device,
            logger=logger,
        )


    finally:
        logger.finish()
    
    return computed_metrics
