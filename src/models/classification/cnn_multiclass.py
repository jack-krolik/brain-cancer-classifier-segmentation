import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data.classification import TumorClassificationDataset, DataSplit
import torchmetrics
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassJaccardIndex,
)


DIM = 256


class ClassificationMulticlassCNN(nn.Module):
    def __init__(self):
        super(ClassificationMulticlassCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, 4)

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

train_dataset = TumorClassificationDataset(
    root_dir="datasets/",
    split=DataSplit.TRAIN,
    transform=transform,
)

test_dataset = TumorClassificationDataset(
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
soft_metrics = torchmetrics.MetricCollection(
    [
        MulticlassAUROC(4).to(device),
        MulticlassJaccardIndex(4).to(device),
        MulticlassAccuracy(4).to(device),
        MulticlassF1Score(4).to(device),
        MulticlassPrecision(4).to(device),
        MulticlassRecall(4).to(device),
    ]
)
onehot_metrics = torchmetrics.MetricCollection(
    [
        MulticlassAUROC(4).to(device),
        MulticlassJaccardIndex(4).to(device),
        MulticlassAccuracy(4).to(device),
        MulticlassF1Score(4).to(device),
        MulticlassPrecision(4).to(device),
        MulticlassRecall(4).to(device),
    ]
)
model = ClassificationMulticlassCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).float().view(-1, 1)
        labels = labels.squeeze(1).long()

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
        probs = F.softmax(outputs, dim=1).to(torch.float32)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        soft_computed_metrics = soft_metrics(probs, labels)
        onehot_computed_metrics = onehot_metrics(
            F.one_hot(predicted, num_classes=4).to(torch.float32), labels
        )
        correct += (predicted == labels).sum().item()

    total_metrics_soft = soft_metrics.compute()
    total_metrics_onehot = onehot_metrics.compute()
    print(f"Validation Metrics Softmax: ", total_metrics_soft)
    print(f"Validation Metrics Onehot: ", total_metrics_onehot)

"""
20 epochs, 95.0419527078566% accuracy

Validation Metrics Softmax:  {'MulticlassAUROC': tensor(0.9923, device='cuda:0'), 'MulticlassJaccardIndex': tensor(0.9045, device='cuda:0'), 'MulticlassAccuracy': tensor(0.9483, device='cuda:0'), 'MulticlassF1Score': tensor(0.9483, device='cuda:0'), 'MulticlassPrecision': tensor(0.9495, device='cuda:0'), 'MulticlassRecall': tensor(0.9483, device='cuda:0')}
Validation Metrics Onehot:  {'MulticlassAUROC': tensor(0.9663, device='cuda:0'), 'MulticlassJaccardIndex': tensor(0.9045, device='cuda:0'), 'MulticlassAccuracy': tensor(0.9483, device='cuda:0'), 'MulticlassF1Score': tensor(0.9483, device='cuda:0'), 'MulticlassPrecision': tensor(0.9495, device='cuda:0'), 'MulticlassRecall': tensor(0.9483, device='cuda:0')}
"""
