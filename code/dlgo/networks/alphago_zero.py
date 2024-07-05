from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F


class AlphaGoZeroNet(nn.Module):
    def __init__(self, IMG_SIZE=19, num_planes=17):
        super(AlphaGoZeroNet, self).__init__()
        self.img_size = IMG_SIZE

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=num_planes, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
        )

        self.res3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
        )

        self.res4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
        )

        self.res5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
        )

        self.res6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
        )

        self.res7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
        )

        self.res8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
        )

        self.res9 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
        )

        self.res10 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
        )
        self.res11 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
        )

        self.res12 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
        )

        self.res13 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
        )

        self.res14 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
        )

        self.res15 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
        )

        self.res16 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
        )

        self.res17 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
        )

        self.res18 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
        )

        self.res19 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
        )

        self.res20 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * IMG_SIZE * IMG_SIZE, IMG_SIZE * IMG_SIZE + 1),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1 * IMG_SIZE * IMG_SIZE, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        residual = self.conv1(x)
        x = self.res1(residual)
        x += residual
        residual = F.relu(x)
        x = self.res2(residual)
        x += residual
        residual = F.relu(x)
        x = self.res3(residual)
        x += residual
        residual = F.relu(x)
        x = self.res4(residual)
        x += residual
        residual = F.relu(x)
        x = self.res5(residual)
        x += residual
        residual = F.relu(x)
        x = self.res6(residual)
        x += residual
        residual = F.relu(x)
        x = self.res7(residual)
        x += residual
        residual = F.relu(x)
        x = self.res8(residual)
        x += residual
        residual = F.relu(x)
        x = self.res9(residual)
        x += residual
        residual = F.relu(x)
        x = self.res10(residual)
        x += residual
        residual = F.relu(x)
        x = self.res11(residual)
        x += residual
        residual = F.relu(x)
        x = self.res12(residual)
        x += residual
        residual = F.relu(x)
        x = self.res13(residual)
        x += residual
        residual = F.relu(x)
        x = self.res14(residual)
        x += residual
        residual = F.relu(x)
        x = self.res15(residual)
        x += residual
        residual = F.relu(x)
        x = self.res16(residual)
        x += residual
        residual = F.relu(x)
        x = self.res17(residual)
        x += residual
        residual = F.relu(x)
        x = self.res18(residual)
        x += residual
        residual = F.relu(x)
        x = self.res19(residual)
        x += residual
        residual = F.relu(x)
        x = self.res20(residual)
        x += residual
        x = F.relu(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value
    
    def name(self):
        return 'alphagozeronet'


class AlphaGoZeroMiniNet(nn.Module):
    def __init__(self, IMG_SIZE=19, num_planes=17):
        super(AlphaGoZeroMiniNet, self).__init__()
        self.img_size = IMG_SIZE

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=num_planes, out_channels=192, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
        )

        self.res3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
        )

        self.res4 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
        )

        self.res5 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * IMG_SIZE * IMG_SIZE, IMG_SIZE * IMG_SIZE + 1),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1 * IMG_SIZE * IMG_SIZE, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        residual = self.conv1(x)
        x = self.res1(residual)
        x += residual
        residual = F.relu(x)
        x = self.res2(residual)
        x += residual
        residual = F.relu(x)
        x = self.res3(residual)
        x += residual
        residual = F.relu(x)
        x = self.res4(residual)
        x += residual
        residual = F.relu(x)
        x = self.res5(residual)
        x += residual
        x = F.relu(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value
    
    def name(self):
        return 'alphagozeromininet'