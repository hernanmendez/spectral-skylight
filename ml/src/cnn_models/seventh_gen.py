from re import X
from .model_templates import SkylightData, ModelParent
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeventhGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool2x2 = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(12, 96, 11, stride=(3, 3), padding=(5, 5))
        self.conv2 = nn.Conv2d(96, 384, 7, padding=(3, 3))
        self.conv3 = nn.Conv2d(384, 1024, 3, padding=(1, 1))
        self.conv4 = nn.Conv2d(1024, 1024, 3, padding=(1, 1))
        self.conv5 = nn.Conv2d(1024, 1024, 2, stride=(2, 2), padding=(0, 0))

        self.features1 = nn.Linear(5, 256)
        self.features2 = nn.Linear(256, 512)
        self.features3 = nn.Linear(512, 1024 * 2)
        self.features4 = nn.Linear(1024 * 2, 1024 * 4)

        self.fc1 = nn.Linear(1024 * 2 * 2, 1024 * 4)
        self.fc2 = nn.Linear(1024 * 4 + 1024 * 4, 1024 * 4 * 2)
        self.fc3 = nn.Linear(1024 * 4 * 2, len(SkylightData.wavelengths))

    def forward(self, x, x_feature_branch):
        x = F.relu(self.pool2x2(self.conv1(x)), inplace=True)
        x = F.relu(self.pool2x2(self.conv2(x)), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        x = F.relu(self.pool2x2(self.conv5(x)), inplace=True)

        x_feature_branch = F.relu(self.features1(x_feature_branch), inplace=True)
        x_feature_branch = F.relu(self.features2(x_feature_branch), inplace=True)
        x_feature_branch = F.relu(self.features3(x_feature_branch), inplace=True)
        x_feature_branch = F.relu(self.features4(x_feature_branch), inplace=True)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x), inplace=True)

        x = torch.cat((x, x_feature_branch), dim=1)

        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)

        return x
