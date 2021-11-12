from deeplut.Initializer.BaseInitializer import BaseInitializer
from deeplut.trainer.BaseTrainer import BaseTrainer
from deeplut.nn.utils.truth_table import generate_truth_table
from deeplut.trainer.LagrangeTrainer import LagrangeTrainer
from typing import Optional, TypedDict
import torch
import numpy as np


class Memorize(BaseInitializer):
    counter: np.ndarray
    weight_lookup_table: TypedDict

    def __init__(self, trainer: BaseTrainer, device: Optional[str] = None) -> None:
        super().__init__(trainer, device)

    def _input_to_id(self, inputs: np.array) -> torch.Tensor:
        _inputs = inputs.clone().detach()
        _inputs[_inputs == -1] = 0
        power_arr = 2**torch.arange(_inputs.shape[-1])
        power_arr = power_arr.reshape(-1, 1)
        return (power_arr * _inputs).sum(0)

    def generate_weight_lookup(self) -> None:
        k = self.trainer.k
        kk = self.trainer.kk
        self.weight_lookup_table = dict()
        weights = generate_truth_table(k=kk, tables_count=1, device=self.device)
        inputs = generate_truth_table(k=k, tables_count=1, device=self.device)
        with torch.no_grad():
            for weight in weights.T:
                output = []
                for input in inputs.T:
                    lut = LagrangeTrainer(tables_count=1, k=k, binary_calculations=True,
                                          input_expanded=True, device=self.device)
                    lut.weight.data = weight
                    output.append(lut(input).item())
                self.weight_lookup_table[tuple(output)] = weight.tolist()

    def clear(self) -> None:
        self.counter = torch.zeros(self.trainer.tables_count)

    def forward(self, x, expected):
        with torch.no_grad():
            self.counter += self.trainer(x)
            x.view(-1,self.trainer.k)
            
