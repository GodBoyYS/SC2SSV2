import torch
import torch.nn as nn
from .csbem import ClassSpecificBandEnhancement
from .gbam import GlobalBandAmplification

class SpectralCNN(nn.Module):
    def __init__(self, input_size=200, embedding_dim=64, num_classes=16):
        super(SpectralCNN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.csbem = ClassSpecificBandEnhancement(input_size, num_classes)
        self.gbam = GlobalBandAmplification(input_size)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * (input_size // 4), 512)
        self.fc2 = nn.Linear(512, embedding_dim)

    def forward(self, x, class_labels=None):
        if class_labels is not None:
            w_class = self.csbem(class_labels)
            x = x * w_class.unsqueeze(1)
        w_global = self.gbam(x)
        x = x * w_global.unsqueeze(1)

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

class SpectralCNN2D(nn.Module):
    def __init__(self, height=10, width=20, embedding_dim=64):
        super(SpectralCNN2D, self).__init__()
        self.height = height
        self.width = width

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算池化后的空间维度
        final_height = height // 4  # 两次池化，每次除以 2
        final_width = width // 4    # 两次池化，每次除以 2
        flattened_dim = 64 * final_height * final_width  # 64 是 conv2 的输出通道数

        self.fc1 = nn.Linear(flattened_dim, 256)
        self.fc2 = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x