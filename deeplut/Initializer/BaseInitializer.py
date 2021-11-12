from typing import Optional
from deeplut.trainer.BaseTrainer import BaseTrainer


class BaseInitializer:
    trainer: BaseTrainer
    device: Optional[str]

    def __init__(self, trainer: BaseTrainer, device: Optional[str]) -> None:
        self.trainer = trainer
        self.device = device
