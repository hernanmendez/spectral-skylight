from .model_templates import SkylightData, ModelParent
import torch
import torch.nn as nn
import torch.nn.functional as F


class FifthGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool2x2 = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(12, 96, 11, stride=(3, 3), padding=(5, 5))
        self.conv2 = nn.Conv2d(96, 192, 5, padding=(2, 2))
        self.conv3 = nn.Conv2d(192, 256, 3, padding=(1, 1))
        self.conv4 = nn.Conv2d(256, 512, 3, padding=(1, 1))
        self.conv5 = nn.Conv2d(512, 1024, 3, padding=(1, 1))
        self.conv6 = nn.Conv2d(1024, 1024, 3, padding=(1, 1))
        self.conv7 = nn.Conv2d(1024, 1024, 2, stride=(2, 2), padding=(0, 0))

        # self.dropout = nn.Dropout(p=0.6)
        # self.features1 = nn.Linear(5, 16)
        # self.features2 = nn.Linear(16, 16)
        # self.features3 = nn.Linear(16, 16)
        # self.features4 = nn.Linear(16, 16)
        # self.features5 = nn.Linear(16, 16)

        self.dropout = nn.Dropout()

        self.fc1 = nn.Linear(1024 * 4 * 4, 1024 * 4 * 2)
        self.fc2 = nn.Linear(1024 * 4 * 2, 1024 * 4)
        self.fc3 = nn.Linear(1024 * 4, len(SkylightData.wavelengths))

    def forward(self, x_image_branch):
        x_image_branch = F.relu(self.conv1(x_image_branch), inplace=True)  # 64x64
        x_image_branch = F.relu(self.conv2(x_image_branch), inplace=True)  # 64x64
        x_image_branch = F.relu(
            self.pool2x2(self.conv3(x_image_branch)), inplace=True
        )  # 32x32
        x_image_branch = F.relu(self.conv4(x_image_branch), inplace=True)  # 16x16
        x_image_branch = F.relu(self.conv5(x_image_branch), inplace=True)  # 16x16
        x_image_branch = F.relu(self.conv6(x_image_branch), inplace=True)  # 16x16
        x_image_branch = F.relu(
            self.pool2x2(self.conv7(x_image_branch)), inplace=True
        )  # 4x4

        x_image_branch = torch.flatten(x_image_branch, 1)

        x_image_branch = F.relu(self.fc1(self.dropout(x_image_branch)), inplace=True)
        x_image_branch = F.relu(self.fc2(self.dropout(x_image_branch)), inplace=True)
        x_image_branch = self.fc3(x_image_branch)

        # x_feature_branch = F.relu(self.features1(x_feature_branch), inplace=True)
        # x_feature_branch = F.relu(self.features2(x_feature_branch), inplace=True)
        # x_feature_branch = F.relu(self.features3(x_feature_branch), inplace=True)
        # x_feature_branch = F.relu(self.features4(x_feature_branch), inplace=True)
        # x_feature_branch = F.relu(self.features5(x_feature_branch), inplace=True)

        return x_image_branch
