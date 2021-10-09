import unittest
import deeplut
import torch
import numpy as np
from deeplut.optim import *

class TestOptimWrapper(unittest.TestCase):

    def test_retrieving_parameters(self):
        
        self.assertTrue(True)


#region Description
class SimpleFeedForward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
#endregion

if __name__ == '__main__':
    unittest.main()