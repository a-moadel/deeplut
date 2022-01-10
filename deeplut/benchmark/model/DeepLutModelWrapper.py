from deeplut.nn.Linear import Linear as dLinear
from deeplut.nn.Conv2d import Conv2d as dConv2d
import torch.nn as nn
import torch


class DeepLutModelWrapper(nn.Module):
    def __init__(self, model):
        super(DeepLutModelWrapper, self).__init__()
        self._model = model

    def is_deep_lut_layer(self, layer):
        return isinstance(layer, dLinear) or isinstance(layer, dConv2d)

    def set_memorize_init(self):
        for layer in self._model.layers:
            if isinstance(layer, dLinear):
                layer.trainer.set_memorize_as_initializer()

    def pre_initialize(self):
        for layer in self._model.layers:
            if self.is_deep_lut_layer(layer):
                layer.pre_initialize()

    def update_initialized_weights(self):
        for layer in self._model.layers:
            if self.is_deep_lut_layer(layer):
                layer.update_initialized_weights()

    def set_trainer_paramters(self, input_expanded, binary_calculations):
        for layer in self._model.layers:
            if self.is_deep_lut_layer(layer):
                layer.trainer.set_input_expanded(input_expanded)
                layer.trainer.set_binary_calculations(binary_calculations)

    def forward(
        self, x, targets: torch.Tensor = None, initalize: bool = False
    ):
        for layer in self._model.layers:
            if self.is_deep_lut_layer(layer):
                x = layer(x, targets, initalize)
            else:
                x = layer(x)
        return x
