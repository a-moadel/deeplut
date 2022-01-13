import torch
import torch.nn as nn


class ResidualSign(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResidualSign, self).__init__(*args, **kwargs)
        self.mean = nn.Parameter(torch.tensor(1.0, requires_grad=True))

    def forward(self, input):
        _input = input.clone().clamp(-1, 1)
        _input.data = torch.sign(_input.data)
        return _input * self.mean.abs()
