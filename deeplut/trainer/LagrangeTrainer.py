import torch
from deeplut.nn.utils import truth_table
from deeplut.trainer.BaseTrainer import BaseTrainer
from typing import Optional


class LagrangeTrainer(BaseTrainer):

    tables_count: int
    device: Optional[str]
    truth_table: torch.Tensor

    def __init__(
        self,
        tables_count: int,
        k: int,
        binary_calculations: bool,
        input_expanded: bool,
        device: Optional[str],
    ):
        """Lagrange Approximation is using Lagrange interpolation to represent differentiable look-up tables.

        Args:
            tables_count (int): Number of look up tables to train
            k (int): numper of inputs of each look up table
            binary_calculations (bool): whether to force binary calculations - simulate real look up tabls -
            device (str): device of the output tensor.
        """
        self.tables_count = tables_count
        self.device = device
        self.truth_table = truth_table.generate_truth_table(k, 1, device)

        super(LagrangeTrainer, self).__init__(
            tables_count=self.tables_count,
            k=k,
            binary_calculations=binary_calculations,
            input_expanded=input_expanded,
            device=device,
        )

    def _validate_input(self, input: torch.tensor):
        """validate inputs dim before passing throw LUTs

        Args:
            input (torch.tensor): input from forward function.

        Raises:
            Exception: Invalid input dim
        """
        _rows_count = input.shape[-1]
        _tbl_count = int(_rows_count / self.k)
        if _rows_count % self.k != 0 or _tbl_count != self.tables_count:
            raise Exception("Invalid input dim")

    def _binarize(self, input: torch.tensor) -> torch.tensor:
        """binarize input to simulate real LUTs

        Args:
            input (torch.tensor): Nd tensor

        Returns:
            [torch.tensor]: Same dimensions as input but all values either -1,1 in case binary_calculations = True.
        """
        if self.binary_calculations:
            input.data = torch.sign(input.data)
        return input

    def forward(self, input: torch.tensor):

        if not hasattr(self.weight, "org"):
            self.weight.org = self.weight.data.clone()
        self._validate_input(input)
        input = input.view(-1, self.k, 1)

        input_truth_table = self._binarize(input + self.truth_table)

        reduced_table = self._binarize(input_truth_table.prod(dim=-2))
        reduced_table = reduced_table.view(-1, self.tables_count, self.kk)

        out = reduced_table * self._binarize(self.weight)
        out = self._binarize(out)

        out = self._binarize(out.sum(-1))

        return out
