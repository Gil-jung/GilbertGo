from __future__ import absolute_import

import torch.nn as nn


class AlphaGoPolicyNet(nn.Module):
    def __init__(self, IMG_SIZE=19, num_planes=48):
        super(AlphaGoPolicyNet, self).__init__()
        self.img_size = IMG_SIZE

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=num_planes, out_channels=192, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer9 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer10 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer11 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer12 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=1, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = x.view(-1, self.img_size*self.img_size)
        
        return x
    
    def name(self):
        return 'alphagopolicynet'


class AlphaGoValueNet(nn.Module):
    def __init__(self, IMG_SIZE=19, num_planes=49):
        super(AlphaGoValueNet, self).__init__()
        self.img_size = IMG_SIZE

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=num_planes, out_channels=192, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer9 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer10 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer11 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer12 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer13 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(inplace=True)
        )

        self.layer14 = nn.Sequential(
            nn.Linear(in_features=self.img_size*self.img_size, out_features=256),
            nn.ReLU(inplace=True)
        )

        self.layer15 = nn.Sequential(
            nn.Linear(in_features=256, out_features=1),
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
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = x.view(-1, self.img_size*self.img_size)
        x = self.layer14(x)
        x = self.layer15(x)
        
        return x
    
    def name(self):
        return 'alphagovaluenet'