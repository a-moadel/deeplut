import unittest
import deeplut
from deeplut.optim.OptimWrapper import OptimWrapper
from deeplut.nn.Linear import Linear as dLinear
import torch

from deeplut.trainer.LagrangeTrainer import LagrangeTrainer
from deeplut.mask.MaskExpanded import MaskExpanded


class TestOptimWrapper(unittest.TestCase):
    def test_retrieving_parameters(self):
        model = SimpleFeedForward(2, 1)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
        wrapper = OptimWrapper(optimizer)
        self.assertEqual(list(model.parameters()), wrapper._get_params())

    def test_grad_is_masked(self):
        model = SimpleDLUTModel(2, 1)
        input = torch.rand(2, 2)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
        wrapper = OptimWrapper(optimizer, True)
        output = model(input)
        output.sum().backward()
        wrapper.step()
        self.assertFalse((model.fc1.trainer.weight.grad[:, 0] == 0).all())
        self.assertTrue((model.fc1.trainer.weight.grad[:, 1] == 0).all())
        self.assertTrue((model.fc1.trainer.weight.grad[:, 2] == 0).all())
        self.assertTrue((model.fc1.trainer.weight.grad[:, 3] == 0).all())


# region Description


class SimpleFeedForward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleFeedForward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()


# region Description


class SimpleDLUTModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleDLUTModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = dLinear(
            self.input_size,
            self.hidden_size,
            2,
            True,
            False,
            LagrangeTrainer,
            MaskExpanded,
            False,
        )

    def forward(self, x):
        return self.fc1(x)


# endregion


if __name__ == "__main__":
    unittest.main()
