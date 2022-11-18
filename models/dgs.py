import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

class gamma_change(nn.Module):
    def __init__(self, apply_norm=True):
        super(gamma_change, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(64)
        self.linear = nn.Sequential(nn.Linear(64*64*3, 128),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(128, 32),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(32, 1))
        self.gamma = 0
        self.apply_norm = apply_norm

    def forward(self, x):
        gamma = self.pool(x)
        gamma = self.linear(gamma.view(gamma.size(0), -1))
        gamma = 2*torch.sigmoid(gamma).unsqueeze(2).unsqueeze(3)
        x = self.taylor(x, gamma)
        if self.apply_norm:
            return self.normalize(x)
        return x

    def taylor(self, x, gamma):
        out = 1 + gamma*(x-1) + gamma*(gamma-1)*(x-1).pow(2)/2 + gamma*(gamma-1)*(gamma-2)*(x-1).pow(3)/6
        return out

    def normalize(self, x):
        return (x-x.view(x.shape[0], x.shape[1], -1).mean(-1).unsqueeze(2).unsqueeze(2))/x.view(x.shape[0], x.shape[1], -1).std(-1).unsqueeze(2).unsqueeze(2)
