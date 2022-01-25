import torch
from torch.functional import Tensor
from deeplut.nn.BinarizeSign import BinarizeSign
from deeplut.nn.utils import truth_table
from deeplut.trainer.BaseTrainer import BaseTrainer
from deeplut.initializer.Memorize import Memorize
from typing import Optional


class LagrangeTrainer(BaseTrainer):

    device: Optional[str]
    truth_table: torch.Tensor

    def __init__(
        self,
        tables_count: int,
        k: int,
        binarization_level: int,
        input_expanded: bool,
        device: Optional[str],
    ):
        """Lagrange Approximation is using Lagrange interpolation to represent differentiable look-up tables.

        Args:
            tables_count (int): Number of tables consumers need to train
            k (int): Number of inputs for each table.
            binarization_level (int): which level of binarization is applied, 0 no binarization , 1 only weights binarized , 2 input also, and 3 output also binarized
            input_expanded (bool): If set to True, means all LUT's inputs are considered during calculations , else only the first input will considered and the remaining will be masked.
            device (str): device of the output tensor.
        """
        self.device = device
        self.truth_table = truth_table.generate_truth_table(k, 1, device)
        super(LagrangeTrainer, self).__init__(
            tables_count=tables_count,
            k=k,
            binarization_level=binarization_level,
            input_expanded=input_expanded,
            device=device,
        )
        self.gamma = torch.nn.Parameter(
            torch.tensor(1.0, requires_grad=True)
        )  # scaling factor

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

    def forward(
        self,
        input: torch.tensor,
        targets: torch.tensor = None,
        initalize: bool = False,
    ) -> torch.Tensor:

        if initalize and self.initializer is not None and targets is not None:
            self.initializer.update_counter(input, targets)

        self._validate_input(input)
        input_truth_table = input.view(-1, self.k, 1) * self.truth_table

        if not self.input_expanded:
            input_truth_table *= -1
            reduced_table = input_truth_table[:, 0, :]
        else:
            input_truth_table = 1 + input_truth_table
            reduced_table = input_truth_table.prod(dim=-2)

        reduced_table = reduced_table.view(-1, self.tables_count, self.kk)
        if self.binarization_level > 0:
            _weight = BinarizeSign.apply(self.weight)
        else:
            _weight = self.weight
        if not self.input_expanded:
            _weight = _weight * self.weight_mask

        out = reduced_table * _weight * self.gamma.abs()

        return out.sum(-1)

    # if we have more intitalizers , may be better we introduce builders for each base module , where we all the object creation logic should live.
    def set_memorize_as_initializer(self) -> None:
        initializer = Memorize(self.tables_count, self.k, self.kk, self.device)
        self.set_initializer(initializer=initializer)
