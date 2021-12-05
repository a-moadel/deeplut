import numpy as np
import math
from deeplut.mask.MaskBase import MaskBase


class MaskMinimal(MaskBase):
    def __init__(
        self, k: int, table_input_selections: list, replace: bool = False
    ) -> None:
        """MaskMinimal: Implements minimal mask which tries to use a fewer number of LUTs, ex: if we have 5 inputs and LUT with k=2 this will result in 3 LUTs only needed.
                        For now, it connects the inputs consecutively and the remaining inputs are selected from the beginning.

        Args:
            k (int): k degree of each table
            table_input_selections (list): List of tuple(selector,list(selector)) , where selector is 1d Array that select an element in the input. The second element in the tuple
            represents the elements we select a form to fill in the k-1 inputs for the table.
            replace (bool, optional): Either we select from the list with/out the replacement.
        """
        super().__init__(
            k=k, table_input_selections=table_input_selections, replace=replace
        )

    def build(self) -> np.ndarray:
        """build minimal mask, where connect inputs consecutively to use fewer number of LUTs.

        Returns:
            np.ndarray: expansion mask.
        """
        result = []
        length = len(self.table_input_selections)
        for i in range(0, length, self.k):
            for j in range(self.k):
                table_input_selection = self.table_input_selections[
                    (i + j) % length
                ]
                result.append(table_input_selection[0])
        return np.array(result)

    def get_tables_count(self) -> int:
        input_length = len(self.table_input_selections)
        return int(math.ceil(input_length / self.k))
