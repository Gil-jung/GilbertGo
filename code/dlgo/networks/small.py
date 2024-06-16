from __future__ import absolute_import

import torch
from torch.nn import Linear, Flatten, Conv2d, ZeroPad2d
from torch.nn.functional import relu


def layers():
    return [
        ZeroPad2d(3),
        Conv2d(in_channels=1, out_channels=48, kernel_size=7),
        relu(),

        ZeroPad2d(2),
        Conv2d(48, 32, 5),
        relu(),

        ZeroPad2d(2),
        Conv2d(32, 32, 5),
        relu(),

        ZeroPad2d(2),
        Conv2d(32, 32, 5),
        relu(),

        Flatten(),
        Linear(32*19*19, 512),
        relu(),
    ]