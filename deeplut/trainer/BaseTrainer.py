import torch
from typing import Final


class BaseTrainer(torch.nn.Linear):
    """ This class is base class for trainers. 
    """

    binary_calculations: [bool]
    k: Final[int]
    kk: Final[int]

    def __init__(self, in_features: int, out_features: int, k: int, binary_calculations: bool):
        self.binary_calculations = binary_calculations
        self.k = k
        self.kk = 2 ** k
        super(BaseTrainer, self).__init__(
            in_features=in_features, out_features=out_features, bias=False)
