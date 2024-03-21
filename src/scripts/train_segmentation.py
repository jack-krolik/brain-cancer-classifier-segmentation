import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from torchinfo import summary
import torchmetrics
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryJaccardIndex
from sklearn.model_selection import KFold
from tqdm import trange, tqdm
import argparse
import pathlib
from dotenv import load_dotenv
import wandb
import os

torch.set_printoptions(precision=3, edgeitems=40, linewidth=120, sci_mode=False)

from src.models.segmentation.unet import UNet
from src.data.segmentation import TumorSemanticSegmentationDataset
from src.utils.visualize import show_images_with_masks
from src.utils.transforms import DualInputCompose, DualInputResize, DualInputTransform
from src.utils.config import TrainingConfig, Hyperparameters
from src.utils.wandb import create_wandb_config, verify_wandb_config


# LOGIN TO W&B
load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"), verify=True)

SAVE_MODEL_DIR = pathlib.Path(__file__).parent.parent.parent / 'models'

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
    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Input batch size for training (default: 1)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train (default: 10)')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds for cross validation (default: 5)')
    parser.add_argument('--architecture', type=str, default='unet')
    args = parser.parse_args()


    # TODO: accept more model architectures as input
    assert args.architecture in ['unet'], f'Invalid architecture: {args.architecture}'

    # TODO: add metrics to the configuration
    # create a training configuration
    hyperparams = Hyperparameters(optimizer='SGD', loss_fn='BCEWithLogitsLoss', batch_size=args.batch_size, learning_rate=args.learning_rate, n_epochs=args.epochs, additional_params={
        'momentum': 0.99
    })

    return TrainingConfig(architecture=args.architecture, dataset='base_segmentation', n_folds=args.n_folds, hyperparameters=hyperparams)


def main_train_loop(model: torch.nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, config: dict, metrics: torchmetrics.MetricCollection, device: torch.device):
    """
    Main training loop for the model

    Args:
    - model (torch.nn.Module): the model to train
    - train_dataloader (DataLoader): the training set dataloader
    - val_dataloader (DataLoader): the validation set dataloader
    - optimizer (torch.optim.Optimizer): the optimizer algorithm (e.g. SGD, Adam, etc.)
    - loss_fn (torch.nn.Module): the loss function being optimized
    - metrics (torchmetrics.MetricCollection): the metrics to validate the model performance
    """
    num_epochs = config.n_epochs
    total_steps = len(train_dataloader) + len(val_dataloader)

    model.to(device)

    best_eval_loss = float('inf')
    best_model_path = None
    
    for epoch in range(num_epochs):
        with tqdm(total=total_steps, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            train_loss = train(model, train_dataloader, optimizer, loss_fn, config, device, pbar)

            val_metrics = evaluate(model, val_dataloader, loss_fn, config, metrics, device, pbar)

            metrics_bundled = {f'val_{metric_name}': metric_value for metric_name, metric_value in val_metrics.items()}
            metrics_bundled['train_loss'] = train_loss # add the training loss to the metrics
            wandb.log(metrics_bundled)

            # log metrics to console
            print("\n".join([f"{key}: {value:.4f}" for key, value in metrics_bundled.items()]))

            val_loss = metrics_bundled['val_loss']

            if val_loss < best_eval_loss:

                if best_model_path is not None: # remove the previous best model
                    os.remove(best_model_path)
                
                # save the best model locally
                best_model_path = SAVE_MODEL_DIR / f'best_{config.architecture}_{config.dataset}_fold_{config.fold}_model.h5'
                torch.save(model.state_dict(), best_model_path) 
                best_eval_loss = val_loss

                print(f"New best model saved at {best_model_path}")
    
    # save the best model to wandb
    wandb.save(str(best_model_path)) 

def train(model: torch.nn.Module, train_dataloader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, config: dict, device: torch.device, pbar: tqdm):
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
        model.zero_grad()
        output = model(imgs)
        loss = loss_fn(output, masks)
        loss.backward()
        optimizer.step()
        cumalative_loss += loss.item()

        pbar.set_postfix({'Loss': f'{loss.item():.4f}', "Phase": 'Train'})
        pbar.update()
    
    # TODO: Add a scheduler to adjust learning rate

    train_loss = cumalative_loss / len(train_dataloader)

    return train_loss

def evaluate(model: torch.nn.Module, val_dataloader: DataLoader, loss_fn: torch.nn.Module, config: dict, metrics: torchmetrics.MetricCollection, device: torch.device, pbar: tqdm):
    """
    Evaluate the model on the validation set

    Args:
    - model (torch.nn.Module): the model to evaluate
    - dataloader (DataLoader): the validation set dataloader
    - loss_fn (torch.nn.Module): the loss function to use
    - config (TrainingConfig): the training configuration
    - metrics (torchmetrics.MetricCollection): the metrics to track
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

            computed_metrics = metrics(preds, masks)
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', "Phase": 'Validation'})
            pbar.update()
        
    total_metrics = metrics.compute() # Compute the metrics over the entire validation set
    return {'loss': cumalative_loss / len(val_dataloader), **total_metrics} 

def main():
    # Get the training configuration
    training_config = get_train_config()
    device = training_config.device

    # Define the augmentation pipeline for the dataset
    # TODO: Here is where augmentation should be added to the dataset
    base_transforms = DualInputCompose([
        DualInputResize((320, 320)),
        DualInputTransform(transforms.ToTensor())
    ])

    if training_config.dataset == 'base_segmentation':
        # Create Segmentation Dataset instance
        train_dataset = TumorSemanticSegmentationDataset(root_dir=training_config.dataset_root_dir, split='train', transform=base_transforms)
        test_dataset = TumorSemanticSegmentationDataset(root_dir=training_config.dataset_root_dir, split='test', transform=base_transforms)
    else:
        raise ValueError(f'Invalid dataset: {training_config.dataset}')

    print(f'Train dataset length: {len(train_dataset)}')
    print(f'Test dataset length: {len(test_dataset)}')

    # Create a KFold instance
    kfold = KFold(n_splits=training_config.n_folds, shuffle=True, random_state=training_config.random_state)

    # Define Metrics to track (e.g. accuracy, precision, recall, etc.)
    # NOTE: Metrics are defined in the torchmetrics library however, custom metrics can be created if needed
    metrics = torchmetrics.MetricCollection([
        BinaryAUROC().to(device),
        BinaryJaccardIndex().to(device),
        BinaryAccuracy().to(device),
        BinaryF1Score().to(device),
        BinaryPrecision().to(device),
        BinaryRecall().to(device)
    ])

    # Train the model using k-fold cross validation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
        print(f'Fold {fold}')
        print('---------------------------')

        wandb_config = create_wandb_config(training_config, {'fold': fold})

        with wandb.init(project=os.getenv("WANDB_PROJECT"), config=wandb_config, tags=['segmentation']):
            config = wandb.config
            assert verify_wandb_config(config, training_config), 'W&B config does not match training config'
            
            # TODO: determine how to bundle the dataset into a make method
            # randomly sample train and validation ids from the dataset based on the fold
            train_sampler = SubsetRandomSampler(train_ids)
            val_sampler = SubsetRandomSampler(val_ids)

            train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler)
            val_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=val_sampler) 

            if config.architecture == 'unet':
                model = UNet()
            else:
                raise ValueError(f'Invalid architecture: {config.architecture}')
            
            if config.optimizer == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
            else:
                # NOTE: add more optimizers as needed 
                raise ValueError(f'Invalid optimizer: {config.optimizer}')

            if config.loss_fn == 'BCEWithLogitsLoss':
                loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
            
            # Train the model
            main_train_loop(model, train_dataloader, val_dataloader, optimizer, loss_fn, config, metrics, device) 



if __name__ == '__main__':
    main()
