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

DIM = 256

transform = transforms.Compose(
    [
        transforms.Resize((DIM, DIM)),  # TODO: make this larger
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # stats come from ImageNet
    ]
)


train_dataset = TumorBinaryClassificationDataset(
    root_dir="datasets/",
    split=DataSplit.TRAIN,
    transform=transform,
)

test_dataset = TumorBinaryClassificationDataset(
    root_dir="datasets/",
    split=DataSplit.TEST,
    transform=transform,
)

print("Train dataset length: ", len(train_dataset))
print("Test dataset length: ", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class LogisiticRegression(nn.Module):
    def __init__(self):
        super(LogisiticRegression, self).__init__()
        self.linear = nn.Linear(DIM * DIM * 3, 1)

    def forward(self, x):
        x = x.view(-1, DIM * DIM * 3)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
metrics = torchmetrics.MetricCollection(
    [
        BinaryAUROC().to(device),
        BinaryJaccardIndex().to(device),
        BinaryAccuracy().to(device),
        BinaryF1Score().to(device),
        BinaryPrecision().to(device),
        BinaryRecall().to(device),
    ]
)
model = LogisiticRegression().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)

num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).float().view(-1, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = (outputs.data > 0.5).float()  # Using 0.5 as the threshold
        total += labels.size(0)
        computed_metrics = metrics(predicted.squeeze(), labels)
        correct += (predicted.view(-1) == labels).sum().item()

    total_metrics = metrics.compute()
    print(f"Validation Metrics: ", total_metrics)

"""
WRONG:
Achieved as high as (90% accuracy, can def go higher. Had better results on lower resolution??

CORRECT:
I actually forgot to choose the test set to validate one lmao.

Accuracy for 10 epochs was 84%. Can probably get better.

Final Metrics Binary:
   Validation Metrics:  {'BinaryAUROC': tensor(0.5069, device='cuda:0'), 'BinaryJaccardIndex': tensor(0.6935, device='cuda:0'), 'BinaryAccuracy': tensor(0.6949, device='cuda:0'), 'BinaryF1Score': tensor(0.8190, device='cuda:0'), 'BinaryPrecision': tensor(0.6940, device='cuda:0'), 'BinaryRecall': tensor(0.9989, device='cuda:0')} 
"""
