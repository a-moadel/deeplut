from deeplut.nn.Linear import Linear as dLinear
from deeplut.nn.Conv2d import Conv2d as dConv2d
from deeplut.nn.BinaryConv2d import BinaryConv2d
from deeplut.nn.BinaryLinear import BinaryLinear
import torch.nn as nn
import torch


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self._model = model

    def is_deep_lut_layer(self, layer):
        return isinstance(layer, dLinear) or isinstance(layer, dConv2d)

    def is_binary_layer(self, layer):
        return isinstance(layer, BinaryConv2d) or isinstance(layer, BinaryLinear)

    def set_trainer_paramters(self, input_expanded, binary_calculations):
        for layer in self._model.layers:
            if self.is_deep_lut_layer(layer):
                layer.trainer.set_input_expanded(input_expanded)
                layer.trainer.set_binary_calculations(binary_calculations)

    def binarize_weights(self):
        for layer in self._model.layers:
            if self.is_binary_layer(layer):
                layer.weight.data = torch.sign(layer.weight.data)
            elif self.is_deep_lut_layer(layer):
                layer.trainer.weight.data = torch.sign(
                    layer.trainer.weight.data)

    def forward(
        self, x, targets: torch.Tensor = None, initalize: bool = False
    ):
        for layer in self._model.layers:
            if self.is_deep_lut_layer(layer):
                x = layer(x, targets, initalize)
            else:
                x = layer(x)
        return x
