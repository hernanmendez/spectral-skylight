from .model_templates import SkylightData, ModelParent
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourthGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool3x3 = nn.MaxPool2d(3, 3)
        self.pool2x2 = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(12, 32, 7, padding=(3, 3))
        self.conv2 = nn.Conv2d(32, 64, 5, padding=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, padding=(1, 1))

        self.fc1 = nn.Linear(128 * 8 * 8, 2048)

        self.features1 = nn.Linear(5, 16)
        self.features2 = nn.Linear(16, 32)

        self.fc2 = nn.Linear(2048 + 32, 1024)
        self.fc3 = nn.Linear(1024, len(SkylightData.wavelengths))

    def forward(self, x_image_branch, x_feature_branch):
        # x_image_branch 192x192
        x_image_branch = self.pool3x3(
            F.relu(self.conv1(x_image_branch), inplace=True)
        )  # 64x64
        x_image_branch = self.pool2x2(
            F.relu(self.conv2(x_image_branch), inplace=True)
        )  # 32x32
        x_image_branch = self.pool2x2(
            F.relu(self.conv3(x_image_branch), inplace=True)
        )  # 16x16
        x_image_branch = self.pool2x2(
            F.relu(self.conv4(x_image_branch), inplace=True)
        )  # 8x8

        x_image_branch = torch.flatten(x_image_branch, 1)  # 8192

        x_image_branch = F.relu(self.fc1(x_image_branch), inplace=True)  # 2048

        x_feature_branch = F.relu(self.features1(x_feature_branch), inplace=True)  # 16
        x_feature_branch = F.relu(self.features2(x_feature_branch), inplace=True)  # 32

        x = torch.cat((x_image_branch, x_feature_branch), 1)  # 2048 + 32

        x = F.relu(self.fc2(x), inplace=True)  # 1024
        x = self.fc3(x)

        return x
