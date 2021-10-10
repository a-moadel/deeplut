import unittest
import deeplut
import torch
import numpy as np
from deeplut.nn.utils import rand_in_range_execlude

class TestMaskBuilder(unittest.TestCase):

    def test_mask_builder_normal_scenario(self):
        np.random.seed(0)
        actual = rand_in_range_execlude(range_limit = 5, must_have_val = 2, count = 3, device= 'cpu')
        expected = torch.from_numpy(np.array([3,4,1,2]))
        equal_result = torch.eq(actual, expected)
        self.assertTrue(torch.all(equal_result))
    
    def test_missing_arguments(self):
        with self.assertRaises(TypeError):
            actual = rand_in_range_execlude(must_have_val = 2, count = 3, device= 'cpu')
        with self.assertRaises(TypeError):
            actual = rand_in_range_execlude(range_limit = 5, count = 3, device= 'cpu')
        with self.assertRaises(TypeError):
            actual = rand_in_range_execlude(range_limit = 5, must_have_val = 2, device= 'cpu')
        with self.assertRaises(TypeError):
            actual = rand_in_range_execlude(range_limit = 5, must_have_val = 2, count = 3)
            
if __name__ == '__main__':
    unittest.main()