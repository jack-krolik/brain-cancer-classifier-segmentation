from torch import nn
import torch
import torch.nn.functional as F


class FocalLoss(nn.Module):
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


class BBoxLoss(nn.Module):

    def __init__(self, class_loss_weight=1, reg_loss_weight=0):
        super(BBoxLoss, self).__init__()

        self.class_loss_weight = class_loss_weight
        self.reg_loss_weight = reg_loss_weight

        # self.class_loss = FocalLoss()
        self.class_loss = nn.BCEWithLogitsLoss()
        self.reg_loss = nn.SmoothL1Loss()

    def forward(self, pred_labels, pred_targets, y_labels, y_targets):

        # bbox classification loss
        # class_loss = self.class_loss(pred_labels, y_labels)

        # Calculate the weights for each sample in the batch
        # This is a simple example of inverse frequency weighting
        positive_weight = (y_labels == 0).sum() / y_labels.numel()
        negative_weight = 1 - positive_weight

        # Dynamic weighting for balancing positive and negative classes
        # Adjust the pos_weight parameter for BCEWithLogitsLoss
        class_loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([positive_weight / negative_weight]).to("mps")
        )

        # bbox classification loss
        class_loss = class_loss_fn(pred_labels, y_labels)

        # bbox adjuctment loss
        not_nan_mask = ~torch.isnan(y_targets)
        reg_loss = self.reg_loss(pred_targets[not_nan_mask], y_targets[not_nan_mask])

        # combining losses
        return class_loss * self.class_loss_weight + reg_loss * self.reg_loss_weight
