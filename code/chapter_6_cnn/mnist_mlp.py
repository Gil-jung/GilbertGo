from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

import numpy as np
import matplotlib.pyplot as plt
import PIL


class MNISTDNN(nn.Module):
    def __init__(self, IMG_SIZE=28):
        super(MNISTDNN, self).__init__()
        self.fc1 = nn.Linear(IMG_SIZE*IMG_SIZE, 32)
        self.BN1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.BN1(x)
        x = self.fc2(x)
        x = torch.softmax(x, dim=-1)
        return x


class MNISTCNN(nn.Module):
    def __init__(self, IMG_SIZE=28):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, stride=2)
        self.BN1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 5, stride=2)
        self.BN2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 8, 3, stride=1)
        self.fc = nn.Linear(8*2*2, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.BN1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.BN(2)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(-1, 8*2*2)
        x = self.fc(x)
        x = torch.softmax(x, dim=-1)
        return x


def compute_acc(argmax, y):
    count = 0
    for i in range(len(argmax)):
        if argmax[i] == y[i]:
            count += 1
    return count / len(argmax)


IMG_SIZE = 28
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHES = 30

transforms = Compose([
    ToTensor(),
])

trainset = MNIST('../data/', train=True, transform=transforms, download=True)
testset = MNIST('../data/', train=False, transform=transforms, download=True)

args = {
    'num_workers': 1,
    'batch_size': BATCH_SIZE,
    'shuffle': True,
}

train_loader = DataLoader(trainset, **args)
test_loader = DataLoader(testset, **args)

model = MNISTDNN(IMG_SIZE).cuda()

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
num_params = sum([np.prod(p.size()) for p in model_parameters])
print("number of parameters : {}".format(num_params))

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(NUM_EPOCHES):
    tot_loss = 0.0

    for x, y in train_loader:
        optimizer.zero_grad()
        x = x.cuda().view(-1, IMG_SIZE*IMG_SIZE)
        y_ = model(x)
        loss = loss_fn(y_, y.cuda())
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()

    print("Epoch {}, Loss(train) : {}".format(epoch+1, tot_loss))
    if epoch % 2 == 1:
        x, y = next(iter(test_loader))
        x = x.cuda().view(-1, IMG_SIZE*IMG_SIZE)
        y_ = model(x)
        _, argmax = torch.max(y_, dim=-1)
        test_acc = compute_acc(argmax, y.numpy())

        print("Acc(val) : {}".format(test_acc))

torch.save(model.state_dict(), "../models/DNN.pt")