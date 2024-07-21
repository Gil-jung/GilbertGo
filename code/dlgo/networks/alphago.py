from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F

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
            # nn.Softmax2d()
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


class AlphaGoPolicyResNet(nn.Module):
    def __init__(self, IMG_SIZE=19, num_planes=48):
        super(AlphaGoPolicyResNet, self).__init__()
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
            # nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            # nn.ReLU(inplace=True)
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            # nn.ReLU(inplace=True)
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer9 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            # nn.ReLU(inplace=True)
        )

        self.layer10 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer11 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            # nn.ReLU(inplace=True)
        )

        self.layer12 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=1, kernel_size=1, padding=0, bias=False),
            # nn.Softmax2d()
        )

    def forward(self, x):
        residual = self.layer1(x)
        x = self.layer2(residual)
        x = self.layer3(x)
        x += residual
        residual = F.relu(x)
        x = self.layer4(residual)
        x = self.layer5(x)
        x += residual
        residual = F.relu(x)
        x = self.layer6(residual)
        x = self.layer7(x)
        x += residual
        residual = F.relu(x)
        x = self.layer8(residual)
        x = self.layer9(x)
        x += residual
        residual = F.relu(x)
        x = self.layer10(residual)
        x = self.layer11(x)
        x += residual
        residual = F.relu(x)
        x = self.layer12(residual)
        x = x.view(-1, self.img_size*self.img_size)
        
        return x
    
    def name(self):
        return 'alphagopolicyresnet'


class AlphaGoPolicyMiniResNet(nn.Module):
    def __init__(self, IMG_SIZE=19, num_planes=48):
        super(AlphaGoPolicyMiniResNet, self).__init__()
        self.img_size = IMG_SIZE

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=num_planes, out_channels=128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
        )

        self.res3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
        )

        self.res4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, padding=0, bias=False),
            # nn.Softmax2d()
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
        x = F.relu(x)
        x = self.policy_head(x)
        x = x.view(-1, self.img_size*self.img_size)
        
        return x
    
    def name(self):
        return 'alphagopolicyminiresnet'


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
    

class AlphaGoValueResNet(nn.Module):
    def __init__(self, IMG_SIZE=19, num_planes=49):
        super(AlphaGoValueResNet, self).__init__()
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
            # nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            # nn.ReLU(inplace=True)
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            # nn.ReLU(inplace=True)
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer9 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            # nn.ReLU(inplace=True)
        )

        self.layer10 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        self.layer11 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=192),
            # nn.ReLU(inplace=True)
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
        residual = self.layer1(x)
        x = self.layer2(residual)
        x = self.layer3(x)
        x += residual
        residual = F.relu(x)
        x = self.layer4(residual)
        x = self.layer5(x)
        x += residual
        residual = F.relu(x)
        x = self.layer6(residual)
        x = self.layer7(x)
        x += residual
        residual = F.relu(x)
        x = self.layer8(residual)
        x = self.layer9(x)
        x += residual
        residual = F.relu(x)
        x = self.layer10(residual)
        x = self.layer11(x)
        x += residual
        residual = F.relu(x)
        x = self.layer12(residual)
        x = self.layer13(x)
        x = x.view(-1, self.img_size*self.img_size)
        x = self.layer14(x)
        x = self.layer15(x)
        
        return x
    
    def name(self):
        return 'alphagovalueresnet'
    

class AlphaGoValueMiniResNet(nn.Module):
    def __init__(self, IMG_SIZE=19, num_planes=49):
        super(AlphaGoValueMiniResNet, self).__init__()
        self.img_size = IMG_SIZE

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=num_planes, out_channels=128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
        )

        self.res3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
        )

        self.res4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(inplace=True),
        )

        self.value_head1 = nn.Sequential(
            nn.Linear(in_features=self.img_size*self.img_size, out_features=256),
            nn.ReLU(inplace=True)
        )

        self.value_head2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=1),
            nn.Tanh()
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
        x = self.conv2(residual)
        x = self.conv3(x)
        x = x.view(-1, self.img_size*self.img_size)
        x = self.value_head1(x)
        x = self.value_head2(x)
        
        return x
    
    def name(self):
        return 'alphagovalueminiresnet'