from __future__ import absolute_import

import torch.nn as nn


class Medium(nn.Module):
    def __init__(self, IMG_SIZE=19, num_planes=7):
        super(Medium, self).__init__()
        self.img_size = IMG_SIZE

        self.layer1 = nn.Sequential(
            nn.ZeroPad2d(padding=2),
            nn.Conv2d(in_channels=num_planes, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.ZeroPad2d(padding=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        self.layer5 = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        self.layer6 = nn.Sequential(
            nn.Linear(in_features=64*IMG_SIZE*IMG_SIZE, out_features=512),
            nn.ReLU(inplace=True)
        )

        self.layer7 = nn.Sequential(
            nn.Linear(in_features=512, out_features=IMG_SIZE*IMG_SIZE),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 64*self.img_size*self.img_size)
        x = self.layer6(x)
        x = self.layer7(x)
        return x