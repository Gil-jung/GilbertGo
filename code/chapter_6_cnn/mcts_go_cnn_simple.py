from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os

parent_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
current_path = os.path.dirname(__file__)
np.random.seed(123)  

X = np.load(parent_path + '\\generated_games\\features-200.npy')
Y = np.load(parent_path + '\\generated_games\\labels-200.npy')
samples = X.shape[0]
size = 9
input_shape = (size, size, 1)  # The input data shape is 3-dimensional, we use one plane of a 9x9 board representation.

X = X.reshape(samples, 1, size, size)  # We then reshape our input data accordingly.

train_samples = 10000
X_train, X_test = X[:train_samples], X[train_samples:]
Y_train, Y_test = Y[:train_samples], Y[train_samples:]


class MCTSAgentCNN(nn.Module):
    def __init__(self, IMG_SIZE=9):
        super(MCTSAgentCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)   # The first layer in our network is a Conv2D layer with 32 output filters.
        self.conv2 = nn.Conv2d(32, 64, 3)  # For this layer we choose a 3 by 3 convolutional kernel.
        self.fc1 = nn.Linear(64*5*5, 128)
        self.fc2 = nn.Linear(128, IMG_SIZE*IMG_SIZE)

    def forward(self, x):
        x = self.conv1(x)
        x = F.sigmoid(x)
        x = self.conv2(x)
        x = F.sigmoid(x)
        x = x.view(-1, 64*5*5)  # We then flatten the 3D output of the previous convolutional layer...
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)  # ... and follow up with two more dense layers, as we did in the MLP example.
        x = F.sigmoid(x)
        return x


def compute_acc(argmax, y):
    count = 0
    for i in range(len(argmax)):
        if argmax[i] == y[i]:
            count += 1
    return count / len(argmax)


IMG_SIZE = 9
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHES = 5


class GameDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx]).float()
        label = torch.tensor(self.labels[idx]).float()

        if self.transform:
            feature = self.transform(feature)
        
        return feature, label


trainset = GameDataset(X_train, Y_train)
testset = GameDataset(X_test, Y_test)

args = {
    'num_workers': 0,
    'batch_size': BATCH_SIZE,
    'shuffle': True,
}

train_loader = DataLoader(trainset, **args)
test_loader = DataLoader(testset, **args)

model = MCTSAgentCNN(IMG_SIZE).cuda()
print(model)

optimizer = SGD(model.parameters())
loss_fn = nn.MSELoss()
model.train()

for epoch in range(NUM_EPOCHES):
    tot_loss = 0.0

    for x, y in train_loader:
        optimizer.zero_grad()
        x = x.cuda()
        y_ = model(x)
        loss = loss_fn(y_, y.cuda()) 
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()

    print("Epoch {}, Loss(train) : {}".format(epoch+1, tot_loss))
    if epoch % 2 == 1:
        x, y = next(iter(test_loader))
        x = x.cuda()
        y_ = model(x)
        _, argmax = torch.max(y_, dim=-1)
        _, y_arg = torch.max(y, dim=-1)
        test_acc = compute_acc(argmax, y_arg)

        print("Acc(val) : {}".format(test_acc))

torch.save(model.state_dict(), current_path + "\\models\\MCTSAgentSimpleCNN.pt")

model_test = MCTSAgentCNN(IMG_SIZE).cuda()
model_test.load_state_dict(torch.load(current_path + "\\models\\MCTSAgentSimpleCNN.pt"))
model_test.eval()
x, y = next(iter(test_loader))
x = x.cuda()
y_ = model_test(x)
_, argmax = torch.max(y_, dim=-1)
_, y_arg = torch.max(y, dim=-1)
test_acc = compute_acc(argmax, y_arg)

print("Acc(test) : {}".format(test_acc))