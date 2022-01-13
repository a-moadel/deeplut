from deeplut.nn.BinaryConv2d import BinaryConv2d
from deeplut.nn.BinaryLinear import BinaryLinear
from deeplut.nn.ResidualSign import ResidualSign
import torch.nn as nn


class CNV(nn.Module):
    def __init__(self):
        super(CNV, self).__init__()
        self.layers = nn.Sequential(
            BinaryConv2d(3, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            ResidualSign(),
            BinaryConv2d(64, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            ResidualSign(),
            nn.MaxPool2d(2, 2),

            BinaryConv2d(64, 128, 3, bias=False),
            nn.BatchNorm2d(128),
            ResidualSign(),
            BinaryConv2d(128, 128, 3, bias=False),
            nn.BatchNorm2d(128),
            ResidualSign(),
            nn.MaxPool2d(2, 2),

            BinaryConv2d(128, 256, 3, bias=False),
            nn.BatchNorm2d(256),
            ResidualSign(),
            BinaryConv2d(256, 256, 3, bias=False),
            nn.BatchNorm2d(256),
            ResidualSign(),

            nn.Flatten(),

            BinaryConv2d(256, 512, bias=False),
            nn.BatchNorm1d(512),
            ResidualSign(),
            BinaryConv2d(512, 512, bias=False),
            nn.BatchNorm1d(512),
            ResidualSign(),
            BinaryConv2d(512, 10, bias=False),
            nn.BatchNorm1d(10),
            nn.LogSoftmax()
        )


class LFC(nn.Module):
    def __init__(self):
        super(LFC, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            BinaryLinear(784, 256, bias=False),
            nn.BatchNorm1d(256),
            ResidualSign(),
            BinaryLinear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            ResidualSign(),
            BinaryLinear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            ResidualSign(),
            BinaryLinear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            ResidualSign(),
            BinaryLinear(256, 10, bias=False),
            nn.BatchNorm1d(10),
        )
