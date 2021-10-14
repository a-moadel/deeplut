import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Final
import numpy as np
from deeplut.nn.utils import generate_truth_table, reduce_truth_table


class LagrangeBase(torch.nn.Linear):

    tables_count: Final[int]
    k: Final[int]
    kk: Final[int]
    device: Final[str]
    truth_table: Final[torch.Tensor]
    binary_calculations: Final[bool]

    def __init__(self, tables_count: int, k: int, binary_calculations: bool, device: str):
        self.tables_count = tables_count
        self.k = k
        self.kk = 2 ** k
        self.device = device
        self.binary_calc = binary_calc
        self.truth_table = generate_truth_table(
            self.k, self.tables_count, device)
        super(LagrangeBase, self).__init__(self.kk, self.tables_count)

    def _validate_input(self, input):
        _rows_count = input.shape[-1]
        _tbl_count = int(_rows_count/self.k)
        if _rows_count % self.k != 0 or _tbl_count != self.tables_count:
            raise Exception("Invalid input dim")

    def _binarize(self, input: torch.tensor):
        if self.binary_calculations:
            input.data = torch.sign(input.data)
        return input

    def forward(self, input: torch.tensor):
        if not hasattr(self.weight, "org"):
            self.weight.org = self.weight.data.clone()
        self._validate_input(input)
        input_truth_table = _binarize(input) + self.truth_table
        input_truth_table = _binarize(input_truth_table)

        reduced_table = reduce_truth_table(
            self.k, input_truth_table, self.device)
        reduced_table = _binarize(reduced_table)

        out = _binarize(self.weights) * reduced_table
        out = _binarize(out)

        out = _binarize(out.sum(-1))
        
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            
        out = _binarize(out + _binarize(self.bias))
        return out
