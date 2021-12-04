import numpy as np


class MaskBase:

    k: int
    table_input_selections: list
    replace: bool

    def __init__(
        self, k: int, table_input_selections: list, replace: bool = False
    ) -> None:
        """MaskBuilder servers as basis for building expansion masks the wire from high level layers and learners.

        Args:
            k (int): k degree of each table
            table_input_selections (list): List of tuple(selector,list(selector)) , where selector is 1d Array that select an elemen in the input. The second element in the tuple
            represent the elements we select form to fill in the k-1 inputs for the table.
            replace (bool, optional): Either we select from the list with/out replacement.
        """
        self.k = k
        self.table_input_selections = table_input_selections
        self.replace = replace
        super().__init__()