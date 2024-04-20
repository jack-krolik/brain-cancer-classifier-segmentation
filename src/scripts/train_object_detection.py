from typing import List
from src.data.bbox import BoundingBoxDetectionDataset
import torch
from torch.utils.data import DataLoader
import argparse
import pathlib
from dataclasses import dataclass, field
from src.models.object_detection.efficientdet import EfficientDet, save_checkpoint
from src.utils.bbox import calculate_dataset_iou, generate_anchors
from src.utils.loss_functions import BBoxLoss
from tqdm import tqdm
import numpy as np
import os
import pprint as pp
import json
from sklearn.metrics import roc_auc_score

torch.set_printoptions(precision=3, edgeitems=40, linewidth=120, sci_mode=False)

from src.data.classification import DataSplit
from src.utils.transforms import (
    BBoxAnchorEncode,
    BBoxBaseTransform,
    BBoxCompose,
    BBoxResize,
)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 5
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    dataset_root_dir: str = DATASET_BASE_DIR
    optimizer: str = "Adam"
    anchor_aspect_ratios: List[float] = field(default_factory=lambda: [1.0])
    anchor_scales: List[float] = field(default_factory=lambda: [0.1, 0.175, 0.25])
    anchor_feature_map_sizes: List[int] = field(
        default_factory=lambda: [32, 16, 8, 4, 2]
    )
    pretrained_backbone: bool = True
    image_size: int = 256
    save_dir_path: str = None


def main_train_loop(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    anchors: List[np.array],
    loss_fn: torch.nn.Module,
    device: torch.device,
    n_epochs: int,
    configs: TrainingConfig,
):
    model.to(device)

    # keeping track of best model
    tr_log = {
        "tr_loss": [],
        "val_loss": [],
        "val_IoU": [],
        "configs": configs.__dict__,
    }

    for epoch in range(n_epochs):
        with tqdm(
            total=len(train_dataloader) + len(val_dataloader),
            desc=f"Epoch {epoch+1}/{n_epochs}",
            unit="batch",
        ) as pbar:
            train_loss = train_step(
                model, train_dataloader, loss_fn, optimizer, device, pbar
            )

            # _, train_iou = evaluate(
            #     model, train_dataloader, loss_fn, device, anchors, pbar
            # )

            val_loss, val_iou = evaluate(
                model, val_dataloader, loss_fn, device, anchors, pbar
            )

            tr_log["tr_loss"].append(float(train_loss))
            tr_log["val_loss"].append(float(val_loss))
            tr_log["val_IoU"].append(float(val_iou))

            # checking if model needs to be saved
            if max(tr_log["val_IoU"]) == tr_log["val_IoU"][-1]:

                print(f"New best val iou.")
                save_checkpoint(model, configs.save_dir_path + "/best_model.pt")

            save_checkpoint(model, configs.save_dir_path + "/last_model.pt")

            # save training log
            # with open(configs.save_dir_path + "/tr_log.json", "w") as f:
            #     json.dump(tr_log, f, indent=4)

    return tr_log


def train_step(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pbar: tqdm,
):
    model.train()
    cumalative_loss = 0
    optimizer.zero_grad()

    for imgs, (labels, targets, bboxes) in train_dataloader:

        imgs, labels, targets = imgs.to(device), labels.to(device), targets.to(device)
        model.zero_grad()
        y_hat_labels, y_hat_targets = model(imgs)
        y_hat_labels = y_hat_labels.view(y_hat_labels.shape[0], -1)

        optimizer.zero_grad()
        loss = loss_fn(y_hat_labels, y_hat_targets, labels, targets)
        loss.backward()
        optimizer.step()

        cumalative_loss += loss.item()

        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Phase": "Train"})
        pbar.update()

    return cumalative_loss / len(train_dataloader)


def evaluate(
    model: torch.nn.Module,
    eval_dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    anchors: torch.tensor,
    pbar: tqdm,
):

    scores = []
    true_labels = []
    adjustments = []
    bboxes = []

    model.eval()
    with torch.no_grad():
        cumalative_loss = 0
        for imgs, (labels, targets, true_bboxes) in eval_dataloader:
            imgs, labels, targets = (
                imgs.to(device),
                labels.to(device),
                targets.to(device),
            )
            y_hat_labels_batch, y_hat_targets_batch = model(imgs)
            y_hat_labels_batch = y_hat_labels_batch.view(
                y_hat_labels_batch.shape[0], -1
            )
            loss = loss_fn(y_hat_labels_batch, y_hat_targets_batch, labels, targets)
            cumalative_loss += loss.item()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Phase": "Evaluate"})
            pbar.update()

            # keeping track of predictions
            bboxes += true_bboxes.to("cpu").tolist()
            adjustments += (torch.zeros(y_hat_targets_batch.shape) + anchors).tolist()
            # adjustments += (
            #     np.repeat(np.expand_dims(anchors, axis=0), repeats=4, axis=0)
            # ).tolist()
            scores += y_hat_labels_batch.to("cpu").tolist()
            true_labels += labels.to("cpu").tolist()

    # calculating metrics
    avg_iou = calculate_dataset_iou(
        bboxes=bboxes,
        predictions=adjustments,
        scores=scores,
    )
    auc = roc_auc_score(
        np.reshape(np.array(true_labels), -1), sigmoid(np.reshape(np.array(scores), -1))
    )

    print("AVG IOU: ", avg_iou)
    print("AUC: ", auc)

    return cumalative_loss / len(eval_dataloader), avg_iou


def main():

    # Get the training configuration
    training_config = TrainingConfig(
        batch_size=16,
        learning_rate=0.01,
        num_epochs=50,
        device=get_device(),
        dataset_root_dir=DATASET_BASE_DIR,
        optimizer="SGD",
        anchor_aspect_ratios=[1],
        anchor_scales=[0.1, 0.175, 0.25, 0.5, 0.35],
        # anchor_feature_map_sizes=[32, 16, 8, 4, 2],
        anchor_feature_map_sizes=[32],
        pretrained_backbone=True,
        image_size=256,
        save_dir_path="/Users/dimavremenko/Desktop/AI Class/Project/brain-cancer-classifier"
        + "-segmentation/model_registry/od_experiments/test",
    )

    if not os.path.exists(training_config.save_dir_path):
        os.makedirs(training_config.save_dir_path)
        print(f"Directory '{training_config.save_dir_path}' was created.")

    pp.pprint(training_config.__dict__)

    device = training_config.device

    # Defining Anchors
    anchors_centers, anchors_corners = generate_anchors(
        training_config.image_size,
        scales=training_config.anchor_scales,
        aspect_ratios=training_config.anchor_aspect_ratios,
        feature_map_sizes=training_config.anchor_feature_map_sizes,
    )

    # input transforms
    transform = BBoxCompose(
        [
            BBoxBaseTransform(),
            BBoxResize((training_config.image_size, training_config.image_size)),
            BBoxAnchorEncode(
                anchors_centers, positive_iou_threshold=0.5, min_positive_iou=0.3
            ),
        ]
    )

    # initializing datasets
    tr_dataset = BoundingBoxDetectionDataset(
        root_dir=DATASET_BASE_DIR, split=DataSplit.TRAIN, transform=transform
    )
    val_dataset = BoundingBoxDetectionDataset(
        root_dir=DATASET_BASE_DIR, split=DataSplit.VALIDATION, transform=transform
    )
    test_dataset = BoundingBoxDetectionDataset(
        root_dir=DATASET_BASE_DIR, split=DataSplit.TEST, transform=transform
    )

    # initializing data loaders
    tr_data_loader = DataLoader(
        tr_dataset, batch_size=training_config.batch_size, shuffle=True, num_workers=0
    )
    val_data_loader = DataLoader(
        val_dataset, batch_size=training_config.batch_size, shuffle=True, num_workers=0
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=training_config.batch_size, shuffle=True, num_workers=0
    )

    # initializing model
    model = EfficientDet(
        pretrained_backbone=training_config.pretrained_backbone,
        n_classes=1,
        n_anchors=len(training_config.anchor_aspect_ratios)
        * len(training_config.anchor_scales),
        bifpn_layers=3,
        n_channels=64,
    )

    # initializing loss function
    loss = BBoxLoss()

    # initializing optimizer
    if training_config.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=training_config.learning_rate,
        )
    elif training_config.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config.learning_rate,
        )
    else:
        raise ValueError(f"Invalid optimizer: {training_config.optimizer}")

    # Train Model
    tr_log = main_train_loop(
        model=model,
        train_dataloader=tr_data_loader,
        val_dataloader=val_data_loader,
        optimizer=optimizer,
        loss_fn=loss,
        device=device,
        anchors=anchors_centers,
        n_epochs=training_config.num_epochs,
        configs=training_config,
    )

    print(tr_log)

    # Testing Model


if __name__ == "__main__":
    main()
