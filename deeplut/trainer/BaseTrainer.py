import torch


class BaseTrainer(torch.nn.Linear):
    """ This class is the base class for trainers. provide a consistent interface for different ways of tables approximation.
    """

    binary_calculations: [bool]
    k: int
    kk: int

    def __init__(self, tables_count: int, k: int, binary_calculations: bool, device: str) -> None:
        """ Initalize BaseTrainer common data structure.

        Args:
            tables_count (int): Number of tables consumers need to train 
            k (int): Number of inputs for each table.
            binary_calculations (bool): Whether to force binary calculations - simulate real look up tabls -  
            device (str): device of the output tensor.
        """
        self.binary_calculations = binary_calculations
        self.k = k
        self.kk = 2 ** k
        super(BaseTrainer, self).__init__(
            in_features=self.kk, out_features=tables_count, bias=False, device=device)

    def set_binary_calculations(self,binary_calculations: bool):
        """ binary calculations 

        Args:
            binary_calculations (bool): boolean value of the new binary calculations.
        """
        self.binary_calculations = binary_calculations