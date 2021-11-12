import torch
from deeplut.trainer.BaseTrainer import BaseTrainer
import numpy as np
from deeplut.nn.utils.MaskBuilder import MaskBuilder
from typing import Type


class Linear(torch.nn.Module):

    input_mask: torch.Tensor
    tables_count: int
    in_features: int
    out_features: int
    trainer: BaseTrainer

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: int,
        binary_calculations: bool,
        input_expanded: bool,
        trainer_type: Type[BaseTrainer],
        bias: bool = True,
        device: str = None,
    ) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tables_count = in_features * out_features
        self.k = k
        self.kk = 2 ** k
        self.input_mask = self._input_mask_builder(k, in_features)
        self.trainer = trainer_type(
            tables_count=self.tables_count,
            k=k,
            binary_calculations=binary_calculations,
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
        for out_idx in range(self.out_features):
            for in_idx in range(self.in_features):
                _idx_set = set([in_idx])
                _selection = list(_all_inputs_set - _idx_set)
                result.append((in_idx, _selection))
        return result

    def _input_mask_builder(self, k: int, input_size: int) -> torch.Tensor:
        maskBuilder = MaskBuilder(
            self.k, self._table_input_selections_builder(), True
        )
        return torch.from_numpy(maskBuilder.build_expanded()).long()

    def forward(self, input: torch.Tensor):
        assert len(input.shape) == 2
        batch_size = input.shape[0]
        expanded_input = input[:, self.input_mask]
        output = self.trainer(expanded_input).squeeze()
        output = output.view(batch_size, -1)
        assert output.shape[-1] == self.tables_count
        output = output.view(batch_size, self.out_features, self.in_features)
        output = output.sum(-1)
        if self.bias is not None:
            output = output + self.bias
        return output

    def set_input_expanded(self, input_expanded: bool) -> None:
        """Set the value for input expansion, either we use expanded input for not, using expanded input means we only consider first input for each lut.
           Please note that this not applicable if you are using minimal look up tables setup.

        Args:
            input_expanded (bool): boolean value of the new input_expanded.
        """
        self.trainer.set_input_expanded(input_expanded)
