from numpy.core.fromnumeric import trace
import torch
from deeplut.trainer.BaseTrainer import BaseTrainer
from deeplut.initializer.BaseInitializer import BaseInitializer
import numpy as np
from deeplut.mask.MaskBase import MaskBase
from typing import Type


class Linear(torch.nn.Module):

    input_mask: torch.Tensor
    tables_count: int
    in_features: int
    out_features: int
    trainer: BaseTrainer
    mask_builder_type: Type[MaskBase]
    mask_builder: MaskBase

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: int,
        binarization_level: int,
        input_expanded: bool,
        trainer_type: Type[BaseTrainer],
        mask_builder_type: Type[MaskBase],
        bias: bool = True,
        device: str = None,
    ) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.k = k
        self.kk = 2 ** k
        self.mask_builder_type = mask_builder_type
        self.input_mask = self._input_mask_builder()
        self.tables_count = (
            self.mask_builder.get_tables_count() * self.out_features
        )
        self.trainer = trainer_type(
            tables_count=self.tables_count,
            k=k,
            binarization_level=binarization_level,
            input_expanded=input_expanded,
            device=device,
        )

        self.bias = (
            torch.nn.Linear(1, out_features, device=device).bias
            if bias
            else None
        )

    def _table_input_selections_builder(self) -> np.array:
        _all_inputs_set = set(range(self.in_features))
        result = []
        for in_idx in range(self.in_features):
            _idx_set = set([in_idx])
            _selection = list(_all_inputs_set - _idx_set)
            result.append((in_idx, _selection))
        return result

    def _input_mask_builder(self) -> torch.Tensor:
        result = []
        for _ in range(self.out_features):
            self.mask_builder = self.mask_builder_type(
                self.k, self._table_input_selections_builder(), True
            )
            result.append(self.mask_builder.build())
        return np.concatenate(result)

    def forward(
        self,
        input: torch.Tensor,
        targets: torch.tensor = None,
        initalize: bool = False,
    ):
        assert len(input.shape) == 2
        batch_size = input.shape[0]
        expanded_input = input[:, self.input_mask]
        output = self.trainer(expanded_input, targets, initalize).squeeze()
        output = output.view(batch_size, -1)
        assert output.shape[-1] == self.tables_count
        output = output.view(
            batch_size,
            self.out_features,
            int(self.tables_count / self.out_features),
        )
        output = output.sum(-1)
        if self.bias is not None:
            output = output + self.bias
        return output

    def pre_initialize(self):
        self.trainer.clear_initializion()

    def update_initialized_weights(self):
        self.trainer.update_initialized_weights()
