import torch
import math
import numpy as np
import bisect
import random
import itertools
import copy


class MaskBuilder():
    
    tables_count: int
    k: int
    table_input_selections: list
    replace: bool
    
    def __init__(self, tables_count: int, k: int, table_input_selections: list, replace:bool = False):
        self.tables_count = tables_count
        self.k = k 
        self.table_input_selections = table_input_selections
        self.replace = replace
        assert len(self.table_input_selections) == tables_count
        super().__init__()

    def build(self) -> np.ndarray:
        result = []
        for table_input_selection in self.table_input_selections:
            idx = table_input_selection[0]
            possible_selections = table_input_selection[1]
            result.append(idx)
            ids = np.random.choice(len(possible_selections), self.k-1, replace=self.replace)
            for id in ids:
                result.append(possible_selections[id])
        return np.array(result)
