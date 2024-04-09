import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data.classification import TumorBinaryClassificationDataset, DataSplit
import torchmetrics
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryJaccardIndex,
)


DIM = 256


class ClassificationCNN(nn.Module):
    def __init__(self):
        super(ClassificationCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 64 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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
model = ClassificationCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)

num_epochs = 20
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
        computed_metrics = metrics(predicted.squeeze(), labels)
        total += labels.size(0)
        correct += (predicted.view(-1) == labels).sum().item()

    total_metrics = metrics.compute()
    print(f"Validation Metrics: ", total_metrics)

"""
20 epochs, 99.6186117467582% accuracy!!! First try

Validation Metrics:  {'BinaryAUROC': tensor(0.9967, device='cuda:0'), 'BinaryJaccardIndex': tensor(0.9934, device='cuda:0'), 'BinaryAccuracy': tensor(0.9954, device='cuda:0'), 'BinaryF1Score': tensor(0.9967, device='cuda:0'), 'BinaryPrecision': tensor(1., device='cuda:0'), 'BinaryRecall': tensor(0.9934, device='cuda:0')}
"""
