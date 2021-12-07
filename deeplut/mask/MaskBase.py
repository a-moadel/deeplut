import numpy as np


class MaskBase:

    k: int
    table_input_selections: list
    replace: bool

    def __init__(
        self, k: int, table_input_selections: list, replace: bool = False
    ) -> None:
        """MaskBase represents the base class for mask builders. Mask builders are consumed by high-level layers (Linear, Conv2d .. etc)
           to build a mask that wire layer input the LUT inputs in the trainer.

        Args:
            k (int): k degree of each table
            table_input_selections (list): List of tuple(selector,list(selector)) , where selector is 1d Array that select an element in the input. The second element in the tuple
            represents the elements we select from to fill in the k-1 inputs for the table.
            replace (bool, optional): Either we select from the list with/out a replacement.
        """
        self.k = k
        self.table_input_selections = table_input_selections
        self.replace = replace
        super().__init__()

    def build(self):
        return

    def get_tables_count(self):
        return
