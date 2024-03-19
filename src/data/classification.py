import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.enums import DataSplit


class TumorClassificationDataset(Dataset):
    def __init__(self, root_dir, split: DataSplit, transform=None):
        assert (
            split != DataSplit.VALIDATION
        ), "Validation split not included in tumor classification dataset."
        self.root_dir = os.path.join(root_dir, "tumor-classification", split.lower())
        self.transform = transform
        self.classes = os.listdir(self.root_dir)
        self.classes.sort()  # Ensure consistent class ordering
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {
            idx: cls_name for cls_name, idx in self.class_to_idx.items()
        }
        self.samples = []
        # Iterate over each class directory and collect image paths and their labels
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(".jpg"):
                    self.samples.append(
                        (
                            os.path.join(class_dir, img_name),
                            self.class_to_idx[class_name],
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure image is RGB

        if self.transform:
            image = self.transform(image)

        return image, label


class TumorBinaryClassificationDataset(TumorClassificationDataset):
    def __init__(self, root_dir, split: DataSplit, transform=None):
        super().__init__(root_dir, split, transform)
        self.classes = ["notumor", "tumor"]
        self.parent_idx_to_class = self.idx_to_class
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)

        if self.parent_idx_to_class[label] != "notumor":
            label = self.class_to_idx["tumor"]
        else:
            label = self.class_to_idx["notumor"]

        return img, label
