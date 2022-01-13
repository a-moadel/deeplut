import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryLinear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        super(BinaryLinear, self).__init__(*args, **kwargs)
        self.gamma = nn.Parameter(torch.tensor(1.0, requires_grad=True))

    def forward(self, input):
        
        if not hasattr(self.weight, "org"):
            self.weight.org = self.weight.data.clone()
        _weight = self.weight.clone()
        _weight.data = _weight.data.sign()
        
        return F.linear(input, _weight * self.gamma.abs(), self.bias)
