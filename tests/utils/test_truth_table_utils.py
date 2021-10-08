import unittest
import deeplut
import torch
import numpy as np
from deeplut.nn.utils import *

class TestTruthTableGeneration(unittest.TestCase):

    def test_basic_scenario(self):
        actual = generate_truth_table(2,3,'cpu')
        
        expected = torch.from_numpy(np.array([[-1,-1,1,1],[-1,1,-1,1],
                                              [-1,-1,1,1],[-1,1,-1,1],
                                              [-1,-1,1,1],[-1,1,-1,1]]))
        equal_result = torch.eq(actual, expected)
        self.assertTrue(torch.all(equal_result))
if __name__ == '__main__':
    unittest.main()