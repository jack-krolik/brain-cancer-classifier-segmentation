import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import pathlib
from src.enums import DataSplit


class BoundingBoxDetectionDataset(Dataset):
    def __init__(self, root_dir: pathlib.Path, split: DataSplit, transform=None):
        self.root_dir = root_dir / "tumor-segmentation-boxes" / split.lower()
        self.transform = transform

        # Load annotations
        annotations_path = self.root_dir / "_annotations.coco.json"
        with open(annotations_path, "r") as file:
            self.labels = json.load(file)

        # Map image IDs to file names
        self.image_id_to_file_name = {
            image["id"]: image["file_name"] for image in self.labels["images"]
        }

        # Map image IDs to Bboxes
        self.image_id_to_bbox = {}

        for annotation in self.labels["annotations"]:
            image_id = annotation["image_id"]
            self.image_id_to_bbox[image_id] = np.array(annotation["bbox"])

        # Removing bad samples
        INCORRECTLY_ANNOTATED_IMAGES = {1380}
        for image_id in INCORRECTLY_ANNOTATED_IMAGES:
            if image_id in self.image_id_to_file_name:
                del self.image_id_to_file_name[image_id]

        self.image_ids = list(self.image_id_to_file_name.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, self.image_id_to_file_name[image_id])

        image = Image.open(img_path).convert("RGB")
        targets = self.image_id_to_bbox[image_id]

        if self.transform:
            image, targets = self.transform(image, targets)

        return image, targets
