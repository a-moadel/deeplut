import numpy as np
from deeplut.mask.MaskBase import MaskBase


class MaskMinimal(MaskBase):

    def __init__(
        self, k: int, table_input_selections: list, replace: bool = False
    ) -> None:
        """MaskMinimal servers as basic for building expansion masks that wire from high level layers and learners.

        Args:
            k (int): k degree of each table
            table_input_selections (list): List of tuple(selector,list(selector)) , where selector is 1d Array that select an elemen in the input. The second element in the tuple
            represent the elements we select form to fill in the k-1 inputs for the table.
            replace (bool, optional): Either we select from the list with/out replacement.
        """
        super().__init__(k=k, table_input_selections=table_input_selections, replace=replace)

    def build(self) -> np.ndarray:
        """build minimal mask, where connect inputs concetively to use fewer number of luts.

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
        return int(math.ceil(input_length / self.k)) * self.k