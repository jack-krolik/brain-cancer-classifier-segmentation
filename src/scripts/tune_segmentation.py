from bayes_opt import BayesianOptimization
import os
import numpy as np
from torchvision.transforms import transforms
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryAUROC, Dice
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from src.data.datasets import prepare_datasets, DatasetType
from src.utils.config import TrainingConfig, Hyperparameters
from src.utils.logging import WandbLogger, LocalLogger
from src.trainer import k_fold_cross_validation, train_model
from src.utils.transforms import DualInputCompose, DualInputResize, DualInputTransform
from src.metrics import BinaryIoU
from src.models.model_utils import SegmentationArchitecture

"""
Use Bayesian Optimization to tune the hyperparameters of the segmentation model.
1. Learning rate
2. Batch size
3. Maybe others ...
"""


def run_train(lr: float, batch_size: int, config: TrainingConfig):
    """
    Run the training script with the given hyperparameters

    Args:
    - lr (float): the learning rate
    - batch_size (int): the batch size
    """
    print(f"Running training with lr={lr}, batch_size={batch_size}")

    config.hyperparameters.learning_rate = lr
    config.hyperparameters.batch_size = int(batch_size)

    config = TrainingConfig(**config.__dict__)

    if config.use_wandb:
        logger = WandbLogger(config, os.getenv("WANDB_API_KEY"), project_name=os.getenv("WANDB_PROJECT"), tags=["segmentation", training_config.architecture, training_config.dataset])
    else:
        logger = LocalLogger(config, 'Hyperparameter Tuning UNet')

    base_transforms = DualInputCompose(
        [DualInputResize((320, 320)), DualInputTransform(transforms.ToTensor())]
    )

    train_dataset, test_dataset = prepare_datasets(config, base_transforms)

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
    if config.n_folds > 1:
        k_fold_cross_validation(config, train_dataset, metrics, logger)
    else:
        val_size = 0.2

        train_dataset_size = len(train_dataset)
        train_idx, val_idx = train_test_split(np.arange(train_dataset_size), test_size=val_size, random_state=config.random_state)
        sub_train_dataset = Subset(train_dataset, train_idx)
        val_dataset = Subset(train_dataset, val_idx)
        computed_metrics = train_model(config, sub_train_dataset, val_dataset, metrics, logger)
    
    # want to minimize the val loss and maximize the val iou
    mini = computed_metrics["val_loss"]
    maxi = np.maximum(computed_metrics["val_BinaryIoU"], 1e-10)
    return maxi / mini


def tune_with_bay_opt(config: TrainingConfig):
    """
    Tune the hyperparameters of the segmentation model using Bayesian Optimization

    Args:
    - config (TrainingConfig): the training configuration
    """

    # Define the hyperparameter search space
    pbounds = {
        "lr": (1e-4, 1e-2),
        "batch_size": (4, 16),
    }

    optimizer = BayesianOptimization(
        f=lambda lr, batch_size: run_train(lr, batch_size, config),
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=3,
        n_iter=5,
    )

    print(optimizer.max)
    return optimizer.res

if __name__ == "__main__":
    # Get the training configuration
    hyperparams = Hyperparameters(
        optimizer="SGD",
        loss_fn="BCEWithLogitsLoss",
        batch_size=4,
        learning_rate=1e-3,
        n_epochs=15,
        additional_params={"momentum": 0.99},
    )

    training_config = TrainingConfig(
        architecture= SegmentationArchitecture.UNET,
        dataset=DatasetType.BOX,
        n_folds=2,
        use_wandb=False,
        hyperparameters=hyperparams,
    )

    best_params = tune_with_bay_opt(training_config)

    for p_name, res in best_params.items():
        print(f"{p_name}: {res}")


    