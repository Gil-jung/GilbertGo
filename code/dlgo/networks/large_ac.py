from __future__ import absolute_import

import torch
import torch.nn as nn


class Large_AC(nn.Module):
    def __init__(self, IMG_SIZE=19, num_planes=7, hidden_size=512):
        super(Large_AC, self).__init__()
        self.img_size = IMG_SIZE

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=num_planes, out_channels=64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=48, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(inplace=True)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(inplace=True)
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True)
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True)
        )

        self.layer8 = nn.Sequential(
            nn.Linear(in_features=32*IMG_SIZE*IMG_SIZE, out_features=1024),
            nn.ReLU(inplace=True)
        )

        self.policy_hidden_layer = nn.Sequential(
            nn.Linear(in_features=1024, out_features=hidden_size),
            nn.ReLU(inplace=True)
        )

        self.policy_output_layer = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=IMG_SIZE*IMG_SIZE),
            # nn.Softmax(dim=-1)
        )

        self.value_hidden_layer = nn.Sequential(
            nn.Linear(in_features=1024, out_features=hidden_size),
            nn.ReLU(inplace=True)
        )

        self.value_output_layer = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=1),
            nn.Tanh()
        )


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.view(-1, 32*self.img_size*self.img_size)
        x = self.layer8(x)
        policy = self.policy_hidden_layer(x)
        policy = self.policy_output_layer(policy)
        value = self.value_hidden_layer(x)
        value = self.value_output_layer(value)

        return policy, value
    
    def name(self):
        return 'large_ac'