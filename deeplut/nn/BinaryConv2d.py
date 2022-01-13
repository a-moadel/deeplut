import torch
import torch.nn as nn


class BinaryConv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(BinaryConv2d, self).__init__(*args, **kwargs)
        self.gamma = nn.Parameter(torch.tensor(1.0, requires_grad=True))

    def forward(self, input):
        if not hasattr(self.weight, "org"):
            self.weight.org = self.weight.data.clone()
        _weight = self.weight.clone()
        _weight.data = _weight.data.sign()
        out = self._conv_forward(input, _weight * self.gamma.abs(), self.bias)
        return out