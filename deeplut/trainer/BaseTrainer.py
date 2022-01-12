import torch
from deeplut.initializer.BaseInitializer import BaseInitializer
from typing import Optional


class BaseTrainer(torch.nn.Linear):
    """This class is the base class for trainers. provide a consistent interface for different ways of tables approximation."""

    binary_calculations: bool
    k: int
    kk: int
    input_expanded: bool
    tables_count: int
    initializer: BaseInitializer

    def __init__(
        self,
        tables_count: int,
        k: int,
        binary_calculations: bool,
        input_expanded: bool,
        device: Optional[str],
    ) -> None:
        """Initalize BaseTrainer common data structure.

        Args:
            tables_count (int): Number of tables consumers need to train
            k (int): Number of inputs for each table.
            binary_calculations (bool): Whether to force binary calculations - simulate real look up tabls -
            input_expanded (bool): If set to True, means all LUT's inputs are considered during calculations , else only the first input will considered and the remaining will be masked.
            device (str): device of the output tensor.
        """
        self.k = k
        self.kk = 2 ** k
        self.binary_calculations = binary_calculations
        self.input_expanded = input_expanded
        self.tables_count = tables_count

        super(BaseTrainer, self).__init__(
            in_features=self.kk,
            out_features=tables_count,
            bias=False,
            device=device,
        )
        self.set_input_expanded(input_expanded)

    def set_binary_calculations(self, binary_calculations: bool) -> None:
        """binary calculations

        Args:
            binary_calculations (bool): boolean value of the new binary calculations.
        """
        self.binary_calculations = binary_calculations

    def set_input_expanded(self, input_expanded: bool) -> None:
        """Set the value for input expansion, either we use expanded input for not, using expanded input means we only consider first input for each lut.
           Please note that this not applicable if you are using minimal look up tables setup.

        Args:
            input_expanded (bool): boolean value of the new input_expanded.
        """
        self.input_expanded = input_expanded

        if not self.input_expanded:
            self.weight_mask = torch.zeros_like(self.weight)
            self.weight_mask[:, 0] = 1

    def update_grad_expanded(self) -> None:
        if not self.input_expanded:
            self.weight.grad = self.weight.grad * self.weight_mask

    def set_initializer(self, initializer: BaseInitializer) -> None:
        self.initializer = initializer

    def clear_initializion(self):
        if self.initializer is not None:
            self.initializer.clear()

    def update_initialized_weights(self):
        if self.initializer is not None:
            self.weight.data = self.initializer.update_luts_weights()
