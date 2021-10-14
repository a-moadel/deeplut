import torch
import torch.nn as nn
import torch.nn.functional as F
from deeplut.nn import LagrangeBase
from typing import Final


class Linear(deeplut.nn.LagrangeBase):
    input_mask: Final[torch.tensor]
    tables_count: Final[int]

    def __init__(self, in_features: int, out_features: int, k: int, binary_calculations: bool, device: str):
        self.tables_count = in_features * out_features
        self.input_mask = self._input_mask_builder(k, in_features)
        super(Linear, self).__init__(
            tables_count, k, binary_calculations, device)

    def _input_mask_builder(self, k: int, input_size: int):
        input_mask = torch.randint(0, input_size, (self.tables_count*k,))
        input_mask[::k] = torch.range(0, self.tables_count-1)
        input_mask[::k] %= input_size
        return input_mask

    def forward(self, input: Tensor.torch):
        expanded_input = input[:, input_mask]
        return super().forward(input)