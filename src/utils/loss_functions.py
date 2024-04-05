from torch import nn
import torch


class BBoxLoss(nn.Module):

    def __init__(self, class_loss_weight, reg_loss_weight):
        super(BBoxLoss, self).__init__()

        self.class_loss_weight = class_loss_weight
        self.reg_loss_weight = reg_loss_weight

        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

    def forward(self, pred_labels, pred_targets, y_labels, y_targets):

        # bbox classification loss
        class_loss = self.bce(pred_labels, y_labels)

        # bbox adjuctment loss
        not_nan_mask = ~torch.isnan(y_targets)
        reg_loss = self.mse(pred_targets[not_nan_mask], y_targets[not_nan_mask])

        # combining losses
        return class_loss * self.class_loss_weight + reg_loss * self.reg_loss_weight
