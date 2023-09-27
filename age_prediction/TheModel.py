from DatasetH import AgeDetection
import torch
import pandas as pd
import os
import skimage as io
from torch.utils.data import Dataset
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import (
    DataLoader, )



class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(32 * 16 * 16, 128)  # New fully connected layer
        self.fc2 = nn.Linear(128, num_classes)  # Another fully connected layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

in_channel = 1
num_clusses = 3
learning_rate = 0.001

batch_size = 32
num_epochs = 9
load_model = False

transform = transforms.Compose(
    [transforms.Resize((128, 128)), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
dataset = AgeDetection(csv_file="C:/Bachelor Of Science/ImageClassification/Age detection intenet model/train.csv",
                       root_dir="C:/Bachelor Of Science/ImageClassification/Age detection intenet model/Train",
                       transform=transform)
print(dataset)
total_length = AgeDetection.__len__(dataset)
train_length = int(0.8 * total_length)  # 80% for training, adjust as needed
test_length = total_length - train_length
train_set, test_set = torch.utils.data.random_split(dataset, [train_length, test_length])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

model = CNN().to(device)
critersion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = model(data)
        loss = critersion(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}")

    model.train()
    return num_correct / num_samples


print(f"Accuracy on training set: {check_accuracy(train_loader, model) * 100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model) * 100:.2f}")
