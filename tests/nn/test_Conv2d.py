import torch
import numpy as np
from deeplut.nn import Conv2d
from deeplut.trainer import LagrangeTrainer
import unittest


class test_Conv2d(unittest.TestCase):
    
    def test_get_conv_index_start_at_dilation_1_1(self):
        np.random.seed(0)
        conv2d = Conv2d(in_channels=2, out_channels=3, kernel_size=(3, 3), stride=(
            2, 2), padding=False, dilation=(1, 1), groups=1, bias=True, input_dim=5, trainer_type=LagrangeTrainer)
        actual_conv_start_0_0 = conv2d.get_conv_index_start_at(0, 0)
        expected_conv_start_0_0 = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 2, 0), (0, 2, 1), (0, 2, 2),
                                   (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 2, 0), (1, 2, 1), (1, 2, 2)]

        self.assertEqual(actual_conv_start_0_0, expected_conv_start_0_0)
        
        actual_conv_start_2_2 = conv2d.get_conv_index_start_at(2, 1)
        expected_conv_start_2_2 = [(0, 2, 1), (0, 2, 2), (0, 2, 3), (0, 3, 1), (0, 3, 2), (0, 3, 3), (0, 4, 1), (0, 4, 2), (0, 4, 3),
                                   (1, 2, 1), (1, 2, 2), (1, 2, 3), (1, 3, 1), (1, 3, 2), (1, 3, 3), (1, 4, 1), (1, 4, 2), (1, 4, 3)]

        self.assertEqual(actual_conv_start_2_2, expected_conv_start_2_2)

    def test_get_conv_index_start_at_dilation_1_2(self):
        np.random.seed(0)
        conv2d = Conv2d(in_channels=2, out_channels=3, kernel_size=(3, 3), stride=(
            2, 2), padding=False, dilation=(1, 2), groups=1, bias=True, input_dim=5, trainer_type=LagrangeTrainer)
        actual_conv_start_0_0 = conv2d.get_conv_index_start_at(0, 0)
        expected_conv_start_0_0 = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 1, 0), (0, 1, 2), (0, 1, 4), (0, 2, 0), (0, 2, 2), (0, 2, 4),
                                   (1, 0, 0), (1, 0, 2), (1, 0, 4), (1, 1, 0), (1, 1, 2), (1, 1, 4), (1, 2, 0), (1, 2, 2), (1, 2, 4)]

        self.assertEqual(actual_conv_start_0_0, expected_conv_start_0_0)
        
        actual_conv_start_2_2 = conv2d.get_conv_index_start_at(2, 1)
        expected_conv_start_2_2 = []

        self.assertEqual(actual_conv_start_2_2, expected_conv_start_2_2)

if __name__ == '__main__':
    unittest.main()
