from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

import numpy as np


class MNISTDNN(nn.Module):
    def __init__(self, IMG_SIZE=28):
        super(MNISTDNN, self).__init__()
        self.fc1 = nn.Linear(IMG_SIZE*IMG_SIZE, 392)
        self.fc2 = nn.Linear(392, 196)
        self.fc3 = nn.Linear(196, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x


def compute_acc(argmax, y):
    count = 0
    for i in range(len(argmax)):
        if argmax[i] == y[i]:
            count += 1
    return count / len(argmax)


IMG_SIZE = 28
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHES = 20

transforms = Compose([
    ToTensor(),
])

trainset = MNIST('../chapter_6_cnn/dataset/', train=True, transform=transforms, download=True)
testset = MNIST('../chapter_6_cnn/dataset/', train=False, transform=transforms, download=True)

args = {
    'num_workers': 0,
    'batch_size': BATCH_SIZE,
    'shuffle': True,
}

train_loader = DataLoader(trainset, **args)
test_loader = DataLoader(testset, **args)

model = MNISTDNN(IMG_SIZE).cuda()
print(model)
# model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# num_params = sum([np.prod(p.size()) for p in model_parameters])
# print("number of parameters : {}".format(num_params))

optimizer = SGD(model.parameters())
loss_fn = nn.MSELoss()

for epoch in range(NUM_EPOCHES):
    tot_loss = 0.0

    for x, y in train_loader:
        optimizer.zero_grad()
        x = x.cuda().view(-1, IMG_SIZE*IMG_SIZE)
        y_ = model(x)
        y = F.one_hot(y, num_classes=y_.shape[1]).float()
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

torch.save(model.state_dict(), "../chapter_6_cnn/models/DNN.pt")