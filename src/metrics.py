from torchmetrics.classification.stat_scores import BinaryStatScores
import torch

class BinaryIoU(BinaryStatScores):
    def compute(self):
        tp, fp, _, fn = self._final_state()
        iou = torch.true_divide(tp, tp + fp + fn)
        return torch.nan_to_num(iou)