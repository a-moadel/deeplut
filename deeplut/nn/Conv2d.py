import torch
from deeplut.trainer.LagrangeTrainer import LagrangeTrainer
from deeplut.trainer.BaseTrainer import BaseTrainer

from deeplut.nn.utils.MaskBuilder import MaskBuilder
import math
from typing import Union, Optional, Type


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

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
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
        trainer_type: Type[BaseTrainer] = LagrangeTrainer,
    ):
        super(Conv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
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
        for output_channel in range(self.out_channels):
            table_input_selections = (
                self._table_input_selections_builder_one_output_channel()
            )
            for table_input_selection in table_input_selections:
                result.append(table_input_selection)
        return result

    def _input_mask_builder(self) -> torch.Tensor:
        selections = self._table_input_selections_builder()
        self.tables_count = len(selections)
        maskBuilder = MaskBuilder(self.k, selections, True)
        return torch.from_numpy(maskBuilder.build_expanded()).long()

    def _hout(self):
        _h_out = (
            self.input_dim[0]
            + 2 * self.padding[0]
            - self.dilation[0] * (self.kernel_size[0] - 1)
            - 1
        ) / self.stride[0]
        return math.floor(_h_out + 1)

    def _wout(self):
        _w_out = (
            self.input_dim[1]
            + 2 * self.padding[1]
            - self.dilation[1] * (self.kernel_size[1] - 1)
            - 1
        ) / self.stride[1]
        return math.floor(_w_out + 1)

    def forward(self, input: torch.Tensor):
        assert len(input.shape) == 4
        batch_size = input.shape[0]
        expanded_input = input[
            :,
            self.input_mask[:, 0],
            self.input_mask[:, 1],
            self.input_mask[:, 2],
        ]
        output = self.trainer(expanded_input).squeeze()
        output = output.view(batch_size, -1)
        assert output.shape[-1] == self.tables_count
        output = output.view(
            batch_size, self.out_channels, self._hout(), self._wout(), -1
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
