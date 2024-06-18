from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F


class Small(nn.Module):
    def __init__(self, IMG_SIZE=19, num_planes=1):
        super(Small, self).__init__()
        self.img_size = IMG_SIZE

        self.conv1 = nn.Conv2d(in_channels=num_planes, out_channels=48, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)

        self.bn1 = nn.BatchNorm2d(num_features=48)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.bn4 = nn.BatchNorm2d(num_features=32)

        self.fc1 = nn.Linear(in_features=32*IMG_SIZE*IMG_SIZE, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=IMG_SIZE*IMG_SIZE)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(-1, 32*self.img_size*self.img_size)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        
        return x


# class Small(nn.Module):
#     def __init__(self, IMG_SIZE=19, num_planes=1):
#         super(Small, self).__init__()
#         self.img_size = IMG_SIZE

#         self.layer1 = nn.Sequential(
#             nn.ZeroPad2d(padding=3),
#             nn.Conv2d(in_channels=num_planes, out_channels=48, kernel_size=7),
#             nn.ReLU(inplace=True)
#         )

#         self.layer2 = nn.Sequential(
#             nn.ZeroPad2d(padding=2),
#             nn.Conv2d(in_channels=48, out_channels=32, kernel_size=5),
#             nn.ReLU(inplace=True)
#         )

#         self.layer3 = nn.Sequential(
#             nn.ZeroPad2d(padding=2),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5),
#             nn.ReLU(inplace=True)
#         )

#         self.layer4 = nn.Sequential(
#             nn.ZeroPad2d(padding=2),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5),
#             nn.ReLU(inplace=True)
#         )

#         self.layer5 = nn.Sequential(
#             nn.Linear(in_features=32*IMG_SIZE*IMG_SIZE, out_features=512),
#             nn.ReLU(inplace=True)
#         )

#         self.layer6 = nn.Sequential(
#             nn.Linear(in_features=512, out_features=IMG_SIZE*IMG_SIZE),
#             nn.Softmax(dim=-1)
#         )
    
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = x.view(-1, 32*self.img_size*self.img_size)
#         x = self.layer5(x)
#         x = self.layer6(x)
#         return x