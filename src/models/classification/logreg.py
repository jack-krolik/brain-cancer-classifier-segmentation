import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryJaccardIndex,
)
from torchvision import transforms
from src.data.classification import TumorBinaryClassificationDataset, DataSplit

class LogisiticRegression(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LogisiticRegression, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x
