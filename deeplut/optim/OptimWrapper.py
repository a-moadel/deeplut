import torch


class OptimWrapper:

    optimizer: torch.optim.Optimizer

    BinaryOptim: bool

    def __init__(
        self, optimizer: torch.optim.Optimizer, BinaryOptim: bool = False
    ):
        """constructor function intialize the inner optimizer and set binary optimization mode.

        Args:
            optimizer (torch.optim.Optimizer): pytorch optimizer or an implementation for torch.optim.Optimizer
            BinaryOptim (bool, optional): This Parameter used is used to dis/enable binary optimization. Defaults to False.
        """
        self.optimizer = optimizer
        self.BinaryOptim = BinaryOptim

    def _pre_step(self):
        for p in list(self._get_params()):
            if hasattr(p, "org"):
                p.data.copy_(p.org)

    def _post_step(self):
        for p in list(self._get_params()):
            if hasattr(p, "org"):
                p.org.copy_(p.data.clamp_(-1, 1))
            if hasattr(p, "weight_mask") and p.weight_mask is not None:
                # if input is not expanded we need to make sure grad is masked.
                p.grad *= p.weight_mask

    def _get_params(self):
        return self.optimizer.param_groups[0]["params"]

    def step(self, closure=None):
        """[summary]

        Args:
            closure ([type], optional): [description]. Defaults to None.
        """
        if self.BinaryOptim:
            self._pre_step()

        self.optimizer.step(closure)

        if self.BinaryOptim:
            self._post_step()
