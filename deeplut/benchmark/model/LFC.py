import torch.nn as nn
from torch.nn import Linear


class LFC(nn.Module):
    def __init__(self, k, input_expanded, mask_builder_type, device):
        super(LFC, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            Linear(256, 10),
            nn.BatchNorm1d(10),
        )
