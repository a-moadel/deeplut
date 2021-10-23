import torch
import torch.nn as nn
import torch.nn.functional as F
from deeplut.trainer import LagrangeTrainer, BaseTrainer
import numpy as np


class Conv2d(torch.nn.Module):

    in_channels: int
    out_channels: int
    kernel_size: tuple
    stride: tuple
    padding: bool
    dilation: tuple
    groups: tuple
    bias: bool
    padding_mode: str
    input_dim: tuple
    device: str

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int or tuple, stride: int or tuple = 1, padding: bool = False, dilation: int or tuple = 1, groups: int or tuple = 1, bias: bool = True, padding_mode: str = 'zeros', k: int = 2, binary_calculations: bool = False, input_dim: int or tuple = None, device: str = None, trainer_type: BaseTrainer = None):
        super(Conv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)
        self.padding = padding
        self.dilation = torch.nn.modules.utils._pair(dilation)
        self.groups = torch.nn.modules.utils._pair(groups)
        self.bias = bias
        self.padding_mode = padding_mode
        self.input_dim = torch.nn.modules.utils._pair(input_dim)
        self.device = device
        self.tables_count = self.get_tables_count()
        self.trainer = trainer_type(tables_count=self.tables_count, k=k,
                                    binary_calculations=binary_calculations, device=device)

    def get_conv_index_start_at(self, row_start_idx: int, col_start_idx: int):
        needs_padding = False
        conv_ids = []
        for channel_id in range(self.in_channels):
            for row_idx in range(self.kernel_size[0]):
                shift_in_rows = row_idx*self.dilation[0]
                current_row = row_start_idx + shift_in_rows
                if current_row >= self.input_dim[0]:
                    needs_padding = True
                for col_idx in range(self.kernel_size[1]):
                    shift_in_cols = col_idx*self.dilation[1]
                    currnet_col = col_start_idx + shift_in_cols
                    if currnet_col >= self.input_dim[1]:
                        needs_padding = True
                    
                    conv_ids.append((channel_id, current_row, currnet_col))
        if (not needs_padding) or (self.padding):
            return conv_ids
        return []

    def get_tables_count(self):
        return 1