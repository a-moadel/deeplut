import torch.nn as nn
from deeplut.nn.Conv2d import Conv2d as dConv2d


class CNVLUT(nn.Module):
    def __init__(self, k, trainer, mask_builder_type, device):
        super(CNVLUT, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            dConv2d(256, 256, 3, trainer_type=trainer,
                    mask_builder_type=mask_builder_type, k=k, input_dim=3, device=device),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, 10),
            nn.BatchNorm1d(10),
        )
