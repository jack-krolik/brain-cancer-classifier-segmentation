import os
import torch
import json
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from src.enums import DataSplit

class TumorClassificationDataset(Dataset):
    def __init__(self, root_dir, split: DataSplit, transform=None):
        assert split != DataSplit.VALIDATION, 'Validation split not included in tumor classification dataset.'
        self.root_dir = os.path.join(root_dir, 'tumor-classification', split.lower())
        self.transform = transform
        self.classes = os.listdir(self.root_dir)
        self.classes.sort()  # Ensure consistent class ordering
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        self.samples = []
        # Iterate over each class directory and collect image paths and their labels
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpg'):
                    self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')  # Ensure image is RGB

        if self.transform:
            image = self.transform(image)

        return image, label

class TumorSemanticSegmentationDataset(Dataset):
    def __init__(self, root_dir, split: DataSplit, transform=None):
        self.root_dir = os.path.join(root_dir, 'tumor-segmentation', split.lower())
        self.transform = transform
         # Load annotations
        with open(os.path.join(self.root_dir, '_annotations.coco.json'), 'r') as file:
            self.labels = json.load(file)

        # Map image IDs to file names
        self.image_id_to_file_name = {image['id']: image['file_name'] for image in self.labels['images']}

        # Map image IDs to annotations
        self.image_id_to_annotation = {}
        for annotation in self.labels['annotations']:
            image_id = annotation['image_id']
            # TODO: may need iscrowd field (not sure what it is)
            annotation = {'bbox': np.array(annotation['bbox']), 'segmentation':  annotation['segmentation'][0]}
            if image_id in self.image_id_to_annotation:
                self.image_id_to_annotation[image_id].append(annotation)
            else:
                self.image_id_to_annotation[image_id] = [annotation]


        # THIS IS BAD, but THE DATASET IS ALSO BAD
        INCORRECTLY_ANNOTATED_IMAGES = {1380}
        for image_id in INCORRECTLY_ANNOTATED_IMAGES:
            del self.image_id_to_file_name[image_id]
        
        self.image_ids = list(self.image_id_to_file_name.keys())


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, self.image_id_to_file_name[image_id])
        
        # TODO: determine if this should be RGB or L (grey scale)
        image = Image.open(img_path).convert('RGB')
        annotations = self.image_id_to_annotation[image_id]

        mask = self._create_mask(annotations, image.size)

        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask
    
    def _create_mask(self, annotations, image_size):
        mask = Image.new('L', image_size, 0)
        for annotation in annotations:
            ImageDraw.Draw(mask).polygon(annotation['segmentation'], outline=255, fill=255) # Draw the polygon
        return mask
            