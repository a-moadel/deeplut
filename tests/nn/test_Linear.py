import torch
from typing import Final
import numpy as np
from deeplut.nn import Linear
from deeplut.trainer import LagrangeTrainer
import unittest


class test_Linear(unittest.TestCase):
    def test_mask_creation(self):
        np.random.seed(0)
        linear = Linear(in_features=5, out_features=3, k=2,
                        binary_calculations=False, trainer_type=LagrangeTrainer, device='cpu')
        actual_main_achors = linear.input_mask[::2]
        total_length =5*3*2
        expected_main_achors = np.array(range(15))%5
        self.assertEqual(total_length,linear.input_mask.shape[0])
        self.assertTrue((actual_main_achors==expected_main_achors).all())

    # to be continued
    def test_forward(self):
        np.random.seed(0)
        linear = Linear(in_features=5, out_features=3, k=2,
                        binary_calculations=False, trainer_type=LagrangeTrainer, device='cpu')  
        batch_image = torch.rand(1,5)
        output = linear(batch_image)
                 

if __name__ == '__main__':
    unittest.main()
