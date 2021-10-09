from typing import Final
import torch

class OptimWrapper():

    optimizer: Final[torch.optim.Optimizer]
    BinaryOptim: [bool]

    def __init__(self,  optimizer: torch.optim.Optimizer, BinaryOptim: bool = False):
        self.optimizer = optimizer
        self.BinaryOptim = BinaryOptim

    def __setattr__(self, key, value):
        raise TypeError( "%r is a frozen class" % self )
    
    def preStep(self):
        for p in list(self.getParams()):
            if hasattr(p, "org"):
                p.data.copy_(p.org)

    def postStep(self):
        for p in list(self.getParams()):
            if hasattr(p, "org"):
                p.org.copy_(p.data.clamp_(-1, 1))

    def step(self, closure=None):
        if self.BinaryOptim:
            self.preStep()
        
        self.optimizer(closure)
        
        if self.BinaryOptim:
            self.postStep()

    def getParams(self):
        return self.optimizer.param_groups