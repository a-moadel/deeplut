import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from deeplut.nn.BinarizeSign import BinarizeSign


class BinaryLinear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        super(BinaryLinear, self).__init__(*args, **kwargs)
        stdv = 1 / np.sqrt(self.in_features)
        w = np.random.normal(
            loc=0.0, scale=stdv, size=[self.out_features, self.in_features]
        ).astype(np.float32)
        self.weight = nn.Parameter(torch.tensor(w, requires_grad=True))
        self.gamma = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.binary_training = True

    def forward(self, input):
        if self.binary_training:
            return F.linear(
                input,
                BinarizeSign.apply(self.weight) * self.gamma.abs(),
                self.bias,
            )
        else:
            self.weight.data.clamp_(-1, 1)
            return F.linear(input, self.weight * self.gamma.abs(), self.bias)

    def set_training_parameters(self, binary_training):
        self.binary_training = binary_training
