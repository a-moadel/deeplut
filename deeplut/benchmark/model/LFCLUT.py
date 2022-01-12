import torch.nn as nn
from torch.nn import Linear
from deeplut.nn.Linear import Linear as dLinear


class LFCLUT(nn.Module):
    def __init__(self, k, trainer, mask_builder_type, device):
        super(LFCLUT, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            dLinear(
                256,
                256,
                k=k,
                binary_calculations=True,
                trainer_type=trainer,
                mask_builder_type=mask_builder_type,
                bias=False,
                device=device,
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            dLinear(
                256,
                256,
                k=k,
                binary_calculations=True,
                trainer_type=trainer,
                mask_builder_type=mask_builder_type,
                bias=False,
                device=device,
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            dLinear(
                256,
                256,
                k=k,
                binary_calculations=True,
                trainer_type=trainer,
                mask_builder_type=mask_builder_type,
                bias=False,
                device=device,
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            Linear(256, 10),
            nn.BatchNorm1d(10),
            nn.LogSoftmax()
        )
