from deeplut.nn.BinaryConv2d import BinaryConv2d
from deeplut.nn.BinaryLinear import BinaryLinear
from deeplut.nn.ResidualSign import ResidualSign
from deeplut.nn.Linear import Linear as DLinear
from deeplut.nn.Conv2d import Conv2d as DConv2d

from deeplut.trainer.LagrangeTrainer import LagrangeTrainer
import torch.nn as nn


class CNV(nn.Module):
    def __init__(self, k, mask_builder, device):
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
            DConv2d(
                256,
                256,
                3,
                LagrangeTrainer,
                mask_builder,
                bias=False,
                k=k,
                input_expanded=False,
                input_dim=3,
                device=device,
            ),
            nn.BatchNorm2d(256),
            ResidualSign(),
            nn.Flatten(),
            BinaryLinear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            ResidualSign(),
            BinaryLinear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            ResidualSign(),
            BinaryLinear(512, 10, bias=False),
            nn.BatchNorm1d(10),
        )

    def forward(self, x):
        return self.layers(x)


class LFC(nn.Module):
    def __init__(self, k, mask_builder, device):
        super(LFC, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            BinaryLinear(784, 256, bias=False),
            nn.BatchNorm1d(256),
            ResidualSign(),
            DLinear(
                256,
                256,
                k,
                0,
                False,
                LagrangeTrainer,
                mask_builder,
                False,
                device,
            ),
            nn.BatchNorm1d(256),
            ResidualSign(),
            DLinear(
                256,
                256,
                k,
                0,
                False,
                LagrangeTrainer,
                mask_builder,
                False,
                device,
            ),
            nn.BatchNorm1d(256),
            ResidualSign(),
            DLinear(
                256,
                256,
                k,
                0,
                False,
                LagrangeTrainer,
                mask_builder,
                False,
                device,
            ),
            nn.BatchNorm1d(256),
            ResidualSign(),
            DLinear(
                256,
                10,
                k,
                0,
                False,
                LagrangeTrainer,
                mask_builder,
                False,
                device,
            ),
            nn.BatchNorm1d(10),
        )

    def forward(self, x):
        return self.layers(x)
