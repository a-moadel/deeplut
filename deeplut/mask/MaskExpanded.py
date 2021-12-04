import numpy as np
from deeplut.mask.MaskBase import MaskBase
import math


class MaskExpanded(MaskBase):
    def __init__(
        self, k: int, table_input_selections: list, replace: bool = False
    ) -> None:
        """MaskExpanded servers as basic for building expansion masks that wire from high level layers and learners.

        Args:
            k (int): k degree of each table
            table_input_selections (list): List of tuple(selector,list(selector)) , where selector is 1d Array that select an elemen in the input. The second element in the tuple
            represent the elements we select form to fill in the k-1 inputs for the table.
            replace (bool, optional): Either we select from the list with/out replacement.
        """
        super().__init__(
            k=k, table_input_selections=table_input_selections, replace=replace
        )

    def build(self) -> np.ndarray:
        """build expanded mask where for each input we have a table and remaining inputs for the same table we select in random from the given selection list.

        Returns:
            np.ndarray: expansion mask.
        """
        result = []
        for table_input_selection in self.table_input_selections:
            idx = table_input_selection[0]
            possible_selections = table_input_selection[1]
            result.append(idx)
            ids = np.random.choice(
                len(possible_selections), self.k - 1, replace=self.replace
            )
            for id in ids:
                result.append(possible_selections[id])
        return np.array(result)

    def get_tables_count(self) -> int:
        input_length = len(self.table_input_selections)
        return input_length
