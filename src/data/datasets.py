from enum import auto, StrEnum
from torchvision import transforms as T

from src.data.segmentation import BoxSegmentationDataset, LGGSegmentationDataset
from src.utils.config import TrainingConfig
from src.utils.transforms import ImgOnlyTransform

LGG_NORMALIZE_TRANSFORM = T.Normalize(mean=[0.1047, 0.0965, 0.0985], std=[0.1201, 0.1200, 0.1225]) # See lgg_exploration.ipynb for why these values are derived

class DatasetType(StrEnum):
    BOX = auto()
    LGG = auto()
    # Add more dataset types here


def prepare_datasets(config: TrainingConfig, transforms=None):
    """
    Prepare the datasets for training

    Args:
    - config (TrainingConfig): the training configuration
    - transforms (Optional[Callable]): the transformation to apply to the dataset

    Returns:
    - Tuple[DataLoader, DataLoader]: the training and testing dataloaders
    """
    # Define the augmentation pipeline for the dataset

    # NOTE: ALL AUGMENTATIONS SHOULD BE ADDED HERE
    if transforms is None:
        transforms = DualInputCompose(
            [DualInputTransform(T.ToTensor())]
        )

    if config.dataset == DatasetType.BOX:
        # Create Segmentation Dataset instance
        train_dataset = BoxSegmentationDataset(
            root_dir=config.dataset_root_dir,
            split="train",
            transform=transforms,
        )
        test_dataset = BoxSegmentationDataset(
            root_dir=config.dataset_root_dir,
            split="test",
            transform=transforms,
        )
    elif config.dataset == DatasetType.LGG:
        normalize_transform = ImgOnlyTransform(LGG_NORMALIZE_TRANSFORM)
        
        transforms.add_transform(normalize_transform)

        # Create Segmentation Dataset instance
        train_dataset = LGGSegmentationDataset(
            root_dir=config.dataset_root_dir,
            split="train",
            transform=transforms,
        )
        test_dataset = LGGSegmentationDataset(
            root_dir=config.dataset_root_dir,
            split="test",
            transform=transforms,
        )
    else:
        raise ValueError(f"Invalid dataset: {config.dataset}")

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")
    
    return train_dataset, test_dataset