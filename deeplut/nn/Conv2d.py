import torch
from deeplut.trainer.BaseTrainer import BaseTrainer

from deeplut.mask.MaskBase import MaskBase
import math
from typing import Union, Optional, Type
import numpy as np


class Conv2d(torch.nn.Module):

    in_channels: int
    out_channels: int
    kernel_size: tuple
    stride: tuple
    padding: bool
    dilation: tuple
    groups: tuple
    bias: torch.Tensor
    padding_mode: str
    input_dim: tuple
    device: Optional[str]
    input_mask: torch.Tensor
    k: int
    trainer: BaseTrainer
    mask_builder_type: Type[MaskBase]
    mask_builder: MaskBase
    tables_count: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        trainer_type: Type[BaseTrainer],
        mask_builder_type: Type[MaskBase],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        groups: Union[int, tuple] = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        k: int = 2,
        binary_calculations: bool = True,
        input_expanded: bool = True,
        input_dim: Union[int, tuple] = None,
        device: str = None,
    ):
        super(Conv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.mask_builder_type = mask_builder_type
        self.stride = torch.nn.modules.utils._pair(stride)
        self.padding = torch.nn.modules.utils._pair(padding)
        self.dilation = torch.nn.modules.utils._pair(dilation)
        self.groups = torch.nn.modules.utils._pair(groups)
        self.bias = None
        self.padding_mode = padding_mode
        self.k = k
        self.input_dim = torch.nn.modules.utils._pair(input_dim)
        self.device = device
        self.input_mask = self._input_mask_builder()
        self.tables_count = (
            self.mask_builder.get_tables_count() * self.out_channels
        )
        self.trainer = trainer_type(
            tables_count=self.tables_count,
            k=k,
            binary_calculations=binary_calculations,
            input_expanded=input_expanded,
            device=device,
        )

    def get_conv_index_start_at(self, row_start_idx: int, col_start_idx: int):
        needs_padding = False
        conv_ids = []
        for channel_id in range(self.in_channels):
            for row_idx in range(self.kernel_size[0]):
                shift_in_rows = row_idx * self.dilation[0]
                current_row = row_start_idx + shift_in_rows
                if current_row >= self.input_dim[0]:
                    needs_padding = True
                for col_idx in range(self.kernel_size[1]):
                    shift_in_cols = col_idx * self.dilation[1]
                    currnet_col = col_start_idx + shift_in_cols
                    if currnet_col >= self.input_dim[1]:
                        needs_padding = True
                    conv_ids.append((channel_id, current_row, currnet_col))
        if not needs_padding:
            return conv_ids
        return []

    def _table_input_selections_builder_conv(self, conv_indices):
        _all_inputs_set = set(conv_indices)
        result = []
        for conv_index in conv_indices:
            selections = list(_all_inputs_set - set([conv_index]))
            result.append((conv_index, selections))
        return result

    def _table_input_selections_builder_one_output_channel(self):
        result = []
        for row_idx in range(0, self.input_dim[0], self.stride[0]):
            for col_idx in range(0, self.input_dim[1], self.stride[1]):
                conv_indices = self.get_conv_index_start_at(row_idx, col_idx)
                if len(conv_indices) > 0:
                    table_input_selections = (
                        self._table_input_selections_builder_conv(conv_indices)
                    )
                    for table_input_selection in table_input_selections:
                        result.append(table_input_selection)
        return result

    def _table_input_selections_builder(self):
        result = []
        table_input_selections = (
            self._table_input_selections_builder_one_output_channel()
        )
        for table_input_selection in table_input_selections:
            result.append(table_input_selection)
        return result

    def _input_mask_builder(self) -> torch.Tensor:
        result = []
        for _ in range(self.out_channels):
            selections = self._table_input_selections_builder()
            self.mask_builder = self.mask_builder_type(
                self.k, selections, True
            )
            result.append(self.mask_builder.build())
        return np.concatenate(result)

    def _out_dim(self, dim):
        _out = (
            self.input_dim[dim]
            + 2 * self.padding[dim]
            - self.dilation[dim] * (self.kernel_size[dim] - 1)
            - 1
        ) / self.stride[dim]
        return math.floor(_out + 1)

    def forward(self, input: torch.Tensor, targets: torch.tensor=None, initalize: bool = False):
        assert len(input.shape) == 4
        batch_size = input.shape[0]
        expanded_input = input[
            :,
            self.input_mask[:, 0],
            self.input_mask[:, 1],
            self.input_mask[:, 2],
        ]
        output = self.trainer(expanded_input,targets, initalize).squeeze()
        output = output.view(batch_size, -1)
        assert output.shape[-1] == self.tables_count
        output = output.view(
            batch_size, self.out_channels, self._out_dim(0), self._out_dim(1), -1
        )
        output = output.sum(-1)
        return output

    def set_input_expanded(self, input_expanded: bool) -> None:
        """Set the value for input expansion, either we use expanded input for not, using expanded input means we only consider first input for each lut.
           Please note that this not applicable if you are using minimal look up tables setup.

        Args:
            input_expanded (bool): boolean value of the new input_expanded.
        """
        self.trainer.set_input_expanded(input_expanded)

    def pre_initialize(self):
        self.trainer.clear_initializion()
    
    def update_initialized_weights(self):
        self.trainer.update_initialized_weights()