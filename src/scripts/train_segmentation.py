import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from sklearn.model_selection import KFold
from torchinfo import summary
from tqdm import trange, tqdm
import argparse
import pathlib
from dataclasses import dataclass, field
torch.set_printoptions(precision=3)

from src.models.segmentation.unet import UNet
from src.tumor_dataset import TumorSemanticSegmentationDataset
from src.utils.visualize import show_images_with_masks
from src.utils.transforms import DualInputCompose, DualInputResize, DualInputTransform


"""
Key Notes for Later Improvements / Implementations Details:
- NOTE: UNet paper uses SGD with momentum = 0.99 (this may be a good starting point for hyperparameter tuning)
- NOTE: do we need a scheduler to adjust learning rate?
- NOTE: UNet also introduces pre-computed weights for the loss function to handle class imbalance
- NOTE: BCEWithLogitsLoss is used for binary segmentation tasks (like this one) and is a combination of Sigmoid and BCELoss
- NOTE: Main benefit of BCEWithLogitsLoss is that it is numerically stable because it uses the log-sum-exp trick
"""

DATASET_BASE_DIR = pathlib.Path(__file__).parent.parent.parent / 'datasets'

def get_device():
    """
    Get the device to use for training (cuda if available, then mps, then cpu)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
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
    optimizer: str = 'SGD' 
    momentum: float = 0.99  

def train(model: torch.nn.Module, train_dataloader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, config: TrainingConfig):
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
        with tqdm(total=len(train_dataloader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for batch in train_dataloader:
                imgs, masks = batch

                imgs, masks = imgs.to(device), masks.to(device)
                model.zero_grad()
                output = model(imgs)
                loss = loss_fn(output, masks)
                loss.backward()
                optimizer.step()
                cumalative_loss += loss.item()

                accuracy = (output.round() == masks).float().mean()
                pbar.set_postfix({'Loss': loss.item(), 'Accuracy': accuracy.item()})
                pbar.update()
        print(f'Epoch {epoch+1}/{num_epochs} - Avg Loss: {cumalative_loss / len(train_dataloader)}')

def evaluate(model: torch.nn.Module, val_dataloader: DataLoader, loss_fn: torch.nn.Module, config: TrainingConfig):
    """
    Evaluate the model on the validation set

    Args:
    - model (torch.nn.Module): the model to evaluate
    - dataloader (DataLoader): the validation set dataloader
    - loss_fn (torch.nn.Module): the loss function to use
    """
    device = config['device']

    model.eval()

    with torch.no_grad():
        cumalative_loss = 0
        with tqdm(total=len(val_dataloader), desc='Validation', unit='batch') as pbar:
            for imgs, masks in val_dataloader:
                imgs, masks = imgs.to(device), masks.to(device)
                output = model(imgs)
                loss = loss_fn(output, masks)
                cumalative_loss += loss.item()

                pbar.set_postfix({'Loss': loss.item()})
                pbar.update()

        print(f'Validation Loss: {cumalative_loss / len(val_dataloader)}')

def get_config():
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
    args = parser.parse_args()

    config = TrainingConfig(batch_size=args.batch_size, learning_rate=args.learning_rate, num_epochs=args.epochs)
    # validate the configuration
    assert config.batch_size > 0, 'Batch size must be greater than 0'
    assert 1 > config.learning_rate > 0, 'Learning rate must be greater than 0'
    assert config.num_epochs > 0, 'Number of epochs must be greater than 0'
    assert pathlib.Path(config.dataset_root_dir).exists(), 'Dataset root directory does not exist'
    
    return config

    
def main():
    # Get the training configuration
    config = get_config()
    device = config.device

    # Create Segmentation Dataset instance
    # TODO: Here is where augmentation should be added to the dataset
    base_transforms = DualInputCompose([
        DualInputResize((320, 320)),
        DualInputTransform(transforms.ToTensor())
    ])

    train_dataset = TumorSemanticSegmentationDataset(root_dir=config.dataset_root_dir, split='train', transform=base_transforms)
    test_dataset = TumorSemanticSegmentationDataset(root_dir=config.dataset_root_dir, split='test', transform=base_transforms)

    print(f'Train dataset length: {len(train_dataset)}')
    print(f'Test dataset length: {len(test_dataset)}')

    # Create a KFold instance
    kfold = KFold(n_splits=config.num_folds, shuffle=True, random_state=config.random_state)

    # Train the model using k-fold cross validation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
        print(f'Fold {fold}')
        print('---------------------------')
        
        # randomly sample train and validation ids from the dataset based on the fold
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)

        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler)
        val_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=val_sampler) 

        # Create a UNet model to train on this fold
        model = UNet()
        # summary(model, input_size=(config.batch_size, 3, 320, 320)) # TODO - programatically get input size

        # Move the model to the device
        model.to(device)

        if config.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
        else:
            raise ValueError(f'Invalid optimizer: {config.optimizer}')

        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean') # Binary Cross Entropy with Logits Loss

        # Train the model
        train(model, train_dataloader, optimizer, loss_fn, config)
        
        # Evaluate the model
        evaluate(model, val_dataloader, loss_fn)


if __name__ == '__main__':
    main()
