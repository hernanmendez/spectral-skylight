from .model_templates import SkylightData, ModelParent
import torch
import torch.nn as nn
import torch.nn.functional as F


class ThirdGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3 * 8, 64, 7, padding=(3, 3))  # 100x100
        self.pool2x2 = nn.MaxPool2d(2, 2)  # 50x50
        # nn.ReLU(inplace=True),

        self.conv2 = nn.Conv2d(64, 128, 5, padding=(2, 2))  # 50x50
        # nn.MaxPool2d(2, 2),  # 25, 25
        # nn.ReLU(inplace=True),

        self.conv3 = nn.Conv2d(128, 256, 3, padding=(1, 1))  # 25x25
        self.adaptivepool = nn.AdaptiveMaxPool2d(16)  # 16x16
        # nn.ReLU(inplace=True),

        self.conv4 = nn.Conv2d(256, 512, 2, stride=(2, 2), padding=(0, 0))  # 8x8
        # nn.ReLU(inplace=True),

        self.conv5 = nn.Conv2d(512, 1024, 2, stride=(2, 2), padding=(0, 0))  # 4x4
        # nn.ReLU(inplace=True),

        # nn.Flatten(),

        self.dropout1 = nn.Dropout()
        self.fc1 = nn.Linear(1024 * 4 * 4, 8192)

        self.dropout2 = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(8192, 4096)

        self.features1 = nn.Linear(5, 32)
        self.features2 = nn.Linear(32, 64)

        self.fc3 = nn.Linear(4096 + 64, 2048)
        self.fc4 = nn.Linear(2048, 2048)
        self.fc5 = nn.Linear(2048, len(SkylightData.wavelengths))

    def forward(self, x_image_branch, x_feature_branch):
        x_image_branch = self.pool2x2(F.relu(self.conv1(x_image_branch), inplace=True))
        x_image_branch = self.pool2x2(F.relu(self.conv2(x_image_branch), inplace=True))
        x_image_branch = self.adaptivepool(
            F.relu(self.conv3(x_image_branch), inplace=True)
        )
        x_image_branch = F.relu(self.conv4(x_image_branch), inplace=True)
        x_image_branch = F.relu(self.conv5(x_image_branch), inplace=True)

        x_image_branch = torch.flatten(x_image_branch, 1)
        x_image_branch = F.relu(self.fc1(self.dropout1(x_image_branch)), inplace=True)
        x_image_branch = F.relu(self.fc2(self.dropout2(x_image_branch)), inplace=True)

        x_feature_branch = F.relu(self.features1(x_feature_branch), inplace=True)
        x_feature_branch = F.relu(self.features2(x_feature_branch), inplace=True)

        x = torch.cat((x_image_branch, x_feature_branch), 1)

        x = F.relu(self.fc3(x), inplace=True)
        x = F.relu(self.fc4(x), inplace=True)
        x = F.relu(self.fc5(x), inplace=True)

        return x
