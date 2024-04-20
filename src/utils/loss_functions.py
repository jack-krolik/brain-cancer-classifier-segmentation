from torch import nn
import torch
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Referance: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha=0.01, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute the cross entropy
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Apply the focusing parameter
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return torch.mean(F_loss)
        elif self.reduction == "sum":
            return torch.sum(F_loss)
        else:  # 'none'
            return F_loss


class WeightedBCELoss(nn.Module):
    """
    Formula: L = -w_pos * y * log(sigma(x)) - w_neg * (1 - y) * log(1 - sigma(x))
    Where:
    - L is the loss computed for each batch.
    - w_pos is the weight for positive samples (y=1).
    - w_neg is the weight for negative samples (y=0).
    - y is the label.
    - sigma(x) is the sigmoid function of the predicted logits x.
    - x is the input to the sigmoid function (predicted logits).
    w_pos is calculated as the ratio of negative samples to total samples, and w_neg is 1 minus w_pos.
    """

    def __init__(self):
        super(WeightedBCELoss, self).__init__()

    def forward(self, y_labels, pred_labels):
        positive_weight = (y_labels == 0).sum() / y_labels.numel()
        negative_weight = 1 - positive_weight

        # Calculate the weights for each sample in the batch
        # This is a simple example of inverse frequency weighting
        class_loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([positive_weight / negative_weight]).to(
                y_labels.device
            )
        )

        # bbox classification loss
        return class_loss_fn(pred_labels, y_labels)


class BBoxLoss(nn.Module):

    def __init__(
        self,
        class_loss: str = "weighted_bce",  # options: focal, bce, weighted_bce
        class_loss_weight=1,
        reg_loss="smooth_l1",  # options: mse, smooth_l1
        reg_loss_weight=0,  # currently not training the regressor
    ):
        super(BBoxLoss, self).__init__()

        # initializing the classification loss
        if class_loss == "weighted_bce":
            self.class_loss = nn.BCEWithLogitsLoss()
        elif class_loss == "bce":
            self.class_loss = nn.BCEWithLogitsLoss()
        elif class_loss == "focal":
            self.class_loss = nn.FocalLoss()
        else:
            raise ValueError(f"Invalid classification loss: {class_loss}")

        # initializing the regression loss
        if reg_loss == "mse":
            self.class_loss = nn.MSELoss()
        elif reg_loss == "smooth_l1":
            self.reg_loss = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Invalid classification loss: {class_loss}")

        # weighs of each loss type
        self.class_loss_weight = class_loss_weight
        self.reg_loss_weight = reg_loss_weight

    def forward(self, pred_labels, pred_targets, y_labels, y_targets):

        # bbox classification loss
        class_loss = self.class_loss(pred_labels, y_labels)

        # bbox adjuctment loss (need to filter out nan from negative labeled boxes)
        not_nan_mask = ~torch.isnan(y_targets)
        reg_loss = self.reg_loss(pred_targets[not_nan_mask], y_targets[not_nan_mask])

        # combining losses
        return class_loss * self.class_loss_weight + reg_loss * self.reg_loss_weight
