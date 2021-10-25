import unittest
import deeplut
from deeplut.optim.OptimWrapper import OptimWrapper
import torch


class TestOptimWrapper(unittest.TestCase):
    def test_retrieving_parameters(self):
        model = SimpleFeedForward(2, 1)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
        wrapper = OptimWrapper(optimizer)
        self.assertEqual(list(model.parameters()), wrapper._get_params())


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


# endregion

if __name__ == "__main__":
    unittest.main()
