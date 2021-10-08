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


    def test_odd_scenario_k_0(self):
        actual = generate_truth_table(0,3,'cpu')
        
        expected = torch.from_numpy(np.array([]))
        equal_result = torch.eq(actual, expected)
        self.assertTrue(torch.all(equal_result))

    def test_odd_scenario_tables_count_0(self):
        with self.assertRaises(RuntimeError):
            actual = generate_truth_table(2,0,'cpu')

    def test_missing_argument_tables_count(self):
        with self.assertRaises(TypeError):
            actual = generate_truth_table(0,device='cpu')

    def test_missing_argument_device(self):
        with self.assertRaises(TypeError):
            actual = generate_truth_table(0,1)

    def test_missing_argument_k(self):
        with self.assertRaises(TypeError):
            actual = generate_truth_table(tables_count=1,device='cpu')

if __name__ == '__main__':
    unittest.main()