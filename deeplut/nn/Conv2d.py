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
        binarization_level: int = 0,
        input_expanded: bool = True,
        input_dim: Union[int, tuple] = None,
        device: str = None,
    ):
        super(Conv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)
        self.padding = torch.nn.modules.utils._pair(padding)
        self.dilation = torch.nn.modules.utils._pair(dilation)

        self.mask_builder_type = mask_builder_type

        self.groups = torch.nn.modules.utils._pair(groups)
        self.bias = None
        self.padding_mode = padding_mode
        self.k = k
        self.input_dim = torch.nn.modules.utils._pair(input_dim)
        self.device = device
        self.input_mask = self._input_mask_builder()
        self.tables_count = self.mask_builder.get_tables_count()
        self.trainer = trainer_type(
            tables_count=self.tables_count,
            k=k,
            binarization_level=binarization_level,
            input_expanded=input_expanded,
            device=device,
        )

        self.unfold = torch.nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )

        self.fold = torch.nn.Fold(
            output_size=(self._out_dim(0), self._out_dim(1)),
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )

    def _get_kernel_selections(self, channel_id):
        result = []
        for kh_index in range(self.kernel_size[0]):
            for kw_index in range(self.kernel_size[1]):
                result.append((channel_id, kh_index, kw_index))
        return set(result)

    def _table_input_selections(self):
        result = []
        for out_index in range(self.out_channels):
            for input_index in range(self.in_channels):
                selections = self._get_kernel_selections(input_index)
                for kh_index in range(self.kernel_size[0]):
                    for kw_index in range(self.kernel_size[1]):
                        conv_index = (input_index, kh_index, kw_index)
                        sub_selections = list(selections - set([conv_index]))
                        result.append((conv_index, sub_selections))
        return result

    def _input_mask_builder(self) -> torch.Tensor:
        result = []
        selections = self._table_input_selections()
        self.mask_builder = self.mask_builder_type(self.k, selections, True)
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

    def forward(
        self,
        input: torch.Tensor,
        targets: torch.tensor = None,
        initalize: bool = False,
    ):
        assert len(input.shape) == 4
        batch_size = input.shape[0]
        folded_input = self.unfold(input).transpose(1, 2)
        folded_input = folded_input.view(
            batch_size,
            -1,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        expanded_input = folded_input[
            :,
            :,
            self.input_mask[:, 0],
            self.input_mask[:, 1],
            self.input_mask[:, 2],
        ]
        output = self.trainer(expanded_input, targets, initalize).squeeze()
        output = output.view(
            batch_size,
            self.out_channels,
            self._out_dim(0),
            self._out_dim(1),
            -1,
        ).sum(-1)
        output = output.view(
            batch_size, self._out_dim(0) * self._out_dim(1), -1
        ).transpose(1, 2)
        output = output.view(
            batch_size, self.out_channels, self._out_dim(0), self._out_dim(1)
        )

        return output

    def pre_initialize(self):
        self.trainer.clear_initializion()

    def update_initialized_weights(self):
        self.trainer.update_initialized_weights()
