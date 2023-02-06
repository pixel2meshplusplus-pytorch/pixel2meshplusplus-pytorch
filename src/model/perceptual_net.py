import torch
import torch.nn as nn
from  torch.cuda.amp import autocast


class PerceptualNetwork(nn.Module):
    def __init__(self, input_size=3):
        super(PerceptualNetwork, self).__init__()

        self.conv0_1 = nn.Conv2d(input_size, 16, 3, stride=1, padding=1)
        self.conv0_2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.conv1_1 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  
        self.conv1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(32, 64, 3, stride=2, padding=1) 
        self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(64, 128, 3, stride=2, padding=1) 
        self.conv3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(128, 256, 5, stride=2, padding=2) 
        self.conv4_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(256, 512, 5, stride=2, padding=2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, 3, stride=1, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        with autocast():
            x = self.relu(self.conv0_1(x))
            x = self.relu(self.conv0_2(x))
            x0 = x

            x = self.relu(self.conv1_1(x))
            x = self.relu(self.conv1_2(x))
            x = self.relu(self.conv1_3(x))
            x1 = x

            x = self.relu(self.conv2_1(x))
            x = self.relu(self.conv2_2(x))
            x = self.relu(self.conv2_3(x))
            x2 = x

            x = self.relu(self.conv3_1(x))
            x = self.relu(self.conv3_2(x))
            x = self.relu(self.conv3_3(x))
            x3 = x

            x = self.relu(self.conv4_1(x))
            x = self.relu(self.conv4_2(x))
            x = self.relu(self.conv4_3(x))
            x4 = x

            x = self.relu(self.conv5_1(x))
            x = self.relu(self.conv5_2(x))
            x = self.relu(self.conv5_3(x))
            x = self.relu(self.conv5_4(x))
            x5 = x

        return {
            "geometry_feature": [x0, x1, x2],
            "semantic_feature": [x2, x3, x4, x5]
        }
