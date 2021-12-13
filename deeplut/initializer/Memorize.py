from deeplut.nn.utils.truth_table import generate_truth_table
from deeplut.initializer.BaseInitializer import BaseInitializer
from typing import Dict, Optional
import torch
import numpy as np
import random


class Memorize(BaseInitializer):
    counter: np.ndarray
    weight_lookup_table: Dict

    def __init__(
        self, table_count, k, kk, weight_lookup_table: Dict, device: Optional[str]
    ) -> None:
        super().__init__(table_count, k, kk, weight_lookup_table,device)
        

    def _input_to_id(self, inputs: np.array) -> torch.Tensor:
        # inputs.shape (batch, tablecount * k)
        _inputs = inputs.sign().clone().detach().view(-1, self.k)
        _inputs[_inputs == -1] = 0
        power_arr = 2 ** torch.arange(_inputs.shape[-1]).flip(0).view(-1, 1)

        return (power_arr.T * _inputs).sum(1).view(inputs.shape[0], -1)

    def clear(self) -> None:
        self.counter = torch.zeros(self.table_count, self.kk)

    def _update_counter(self, input_ids: torch.Tensor, target: torch.Tensor) -> None:
        # input_ids.shape(batch, table_count)
        # target.shape (batch,1)
        batch_size = input_ids.shape[0]
        table_count = input_ids.shape[1]

        assert(table_count == self.table_count)

        # convert target from 0,1 => -1,1
        _target_value = (2 * target - 1).flatten()
        # prepare for updating counter.
        _target_value = _target_value.repeat(table_count, 1).T.flatten()

        table_ids = torch.arange(table_count).repeat(batch_size)
        self.counter[table_ids.long(), input_ids.long().flatten()] += _target_value

    def update_counter(self, x: torch.Tensor, target: torch.Tensor) -> None:
        with torch.no_grad():
            input_ids = self._input_to_id(x)
            self._update_counter(input_ids, target)

    def update_luts_weights(self) -> torch.Tensor:
        new_weights = []
        for row in self.counter:
            key = row.detach().cpu().flatten().sign().numpy().tolist() 
            key = self._break_ties_in_key(key)
            value = self.weight_lookup_table[key]
            new_weights.append(value)
        new_weights = torch.tensor(
            new_weights, dtype=torch.float32, requires_grad=True).view(-1, self.kk)
        return new_weights

    def _break_ties_in_key(self,key):
        result = [] #tuple()
        for element in key:
            val = element
            if(val == 0):
                x = random.uniform(0,1)
                if(x<=.5):
                    val = 1
                else:
                    val = -1
            result.append(val)
        return tuple(result)