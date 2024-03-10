import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from torchinfo import summary
import argparse
import os
from tqdm import tqdm

from src.models.segmentation.unet import UNet
from src.tumor_dataset import TumorSemanticSegmentationDataset
from src.utils.visualize import show_images_with_masks

DATASET_ROOT_DIR = 'src/../datasets'


def main():

    # parse args
    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Input batch size for training (default: 1)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train (default: 10)')
    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs

    # Create Segmentation Dataset instance

    # TODO: Here is where augmentation should be added to the dataset
    base_transforms = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])
    # TODO: output masks currently have 1 channel, but the model expects 2 channels
    train_dataset = TumorSemanticSegmentationDataset(root_dir=DATASET_ROOT_DIR, split='train', transform=base_transforms)
    test_dataset = TumorSemanticSegmentationDataset(root_dir=DATASET_ROOT_DIR, split='test', transform=base_transforms)

    print(f'Train dataset length: {len(train_dataset)}')
    print(f'Test dataset length: {len(test_dataset)}')

    # Create a KFold instance
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Train the model
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
        print(f'Fold {fold}')
        print('---------------------------')
        
        # randomly sample train and validation ids from the dataset based on the fold
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler) 

        # Create a UNet model to train on this fold
        model = UNet()
        summary(model, input_size=(batch_size, 3, 320, 320))

        # TODO: Verify best optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # TODO: Verify best loss function (UNet paper uses cross-entropy loss)
        # NOTE: UNet uses cross-entropy loss maybe we should use BCEWithLogitsLoss here
        # NOTE: UNet also introduces pre-computed weights for the loss function to handle class imbalance
        loss_fn = torch.nn.CrossEntropyLoss()

        # Train the model
        model.train()
        cumalative_loss = 0

        for epoch in tqdm(total=epochs):
            print(f'Epoch {epoch}')
            print('---------------------------')
            
            # current batch
            imgs, masks = next(iter(train_dataloader))
            show_images_with_masks(imgs, masks, nmax=batch_size)

            model.zero_grad()
            output = model(imgs)
            loss = loss_fn(output, masks)
            loss.backward()
            optimizer.step()
            cumalative_loss += loss.item()
            print(f'Loss: {loss.item()}')
            
            assert False, 'stop here'
        
        # Validate the model
        model.eval()
        cumalative_loss = 0
        with torch.no_grad():
            for imgs, masks in val_dataloader:
                output = model(imgs)
                loss = loss_fn(output, masks)
                cumalative_loss += loss.item()
        
        print(f'Validation Loss: {cumalative_loss / len(val_dataloader)}')



if __name__ == '__main__':
    main()
