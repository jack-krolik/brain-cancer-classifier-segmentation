import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
    split=DataSplit.TRAIN,
    transform=transform,
)

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
model = LogisiticRegression().to(device)
criterion = nn.BCELoss()
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
        total += labels.size(0)
        correct += (predicted.view(-1) == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the model on test images: {accuracy}%")

"""
Achieved as high as 90% accuracy, can def go higher. Had better results on lower resolution??
"""
