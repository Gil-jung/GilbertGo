from __future__ import absolute_import

import torch.nn as nn


class FullyConnected(nn.Module):
    def __init__(self, IMG_SIZE=19, num_planes=1):
        super(FullyConnected, self).__init__()
        self.img_size = IMG_SIZE

        self.layer1 = nn.Sequential(
            nn.Linear(in_features=num_planes*IMG_SIZE*IMG_SIZE, out_features=128),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Linear(in_features=128, out_features=IMG_SIZE*IMG_SIZE),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x