import torch
import torch.nn as nn
import numpy as np
from deeplut.nn.BinarizeSign import BinarizeSign


class BinaryConv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(BinaryConv2d, self).__init__(*args, **kwargs)
        stdv = 1 / np.sqrt(
            self.kernel_size[0] * self.kernel_size[1] * self.in_channels
        )
        w = np.random.normal(
            loc=0.0,
            scale=stdv,
            size=[
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
            ],
        ).astype(np.float32)
        self.weight = nn.Parameter(torch.tensor(w, requires_grad=True))
        self.gamma = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.binary_training = True

    def forward(self, input):
        if self.binary_training:
            return nn.functional.conv2d(
                input,
                BinarizeSign.apply(self.weight) * self.gamma.abs(),
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        else:
            self.weight.data.clamp_(-1, 1)
            return nn.functional.conv2d(
                input,
                self.weight * self.gamma.abs(),
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    def set_training_parameters(self, binary_training):
        self.binary_training = binary_training
