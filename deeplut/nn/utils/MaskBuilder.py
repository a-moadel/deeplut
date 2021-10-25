import numpy as np


class MaskBuilder:

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

    def build_expanded(self) -> np.ndarray:
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

    def build_minimal(self) -> np.ndarray:
        """[summary]

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
