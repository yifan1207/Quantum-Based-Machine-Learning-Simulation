import torch
import torch.nn as nn 
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self, n):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * n * n // 16, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = x.view(-1, 1, n, n) # reshape the input to a 4D tensor
        x = F.relu(self.conv1(x)) # apply convolutional layer 1
        x = self.pool1(x) # apply pooling layer 1
        x = F.relu(self.conv2(x)) # apply convolutional layer 2
        x = self.pool2(x) # apply pooling layer 2 / pooling layers
        x = x.view(-1, 64 * n * n // 16) # flatten the output
        x = F.relu(self.fc1(x)) # apply fully connected layer 1
        x = F.softmax(self.fc2(x), dim=1) # apply fully connected layer 2 and softmax
        return x
