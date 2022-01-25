import torch
import torch.nn as nn

from deeplut.nn.BinarizeSign import BinarizeSign


class ResidualSign(nn.Module):
    def __init__(self):
        super(ResidualSign, self).__init__()
        self.scaling = nn.Parameter(torch.tensor(1.0, requires_grad=True))

    def forward(self, input):
        return BinarizeSign.apply(input) * self.scaling.abs()
