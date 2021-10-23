import torch
import torch.nn as nn
import torch.nn.functional as F
from deeplut.trainer import LagrangeTrainer, BaseTrainer
import numpy as np


class Linear(torch.nn.Module):

    input_mask: torch.tensor
    tables_count: int
    in_features: int
    out_features: int
    trainer: BaseTrainer

    def __init__(self, in_features: int, out_features: int, k: int, binary_calculations: bool = False, trainer_type: BaseTrainer = None, bias: bool = True, device: str = None) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tables_count = in_features * out_features
        self.k = k
        self.kk = 2**k
        self.input_mask = self._input_mask_builder(k, in_features)
        self.trainer = trainer_type(tables_count=self.tables_count, k=k,
                                    binary_calculations=binary_calculations, device=device)
        self.bias = torch.nn.Linear(
            1, out_features, device=device).bias if bias else None

    def _input_mask_builder_one_layer_random(self) -> np.array:
        _layer_input_size = self.in_features * self.k
        _random_indices = np.random.choice(self.in_features, _layer_input_size)
        _random_indices[::self.k] = np.array(range(self.in_features))
        return _random_indices

    def _input_mask_builder(self, k: int, input_size: int) -> np.array:
        result = []
        for output_id in range(self.out_features):
            result.append(self._input_mask_builder_one_layer_random())
        result = np.array(result)
        return result.flatten()

    def forward(self, input: torch.Tensor):
        assert len(input.shape) == 2
        batch_size = input.shape[0]
        expanded_input = input[:, self.input_mask]
        output = self.trainer(expanded_input).squeeze()
        output = output.view(batch_size, -1)
        assert output.shape[-1] == self.tables_count
        output = output.view(batch_size, self.out_features, self.in_features)
        output = output.sum(-1)
        if self.bias != None:
            output = output+self.bias
        return output
