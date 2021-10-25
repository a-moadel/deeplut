import torch
import numpy as np
from deeplut.nn import Conv2d
from deeplut.trainer import LagrangeTrainer
import unittest


class test_Conv2d(unittest.TestCase):
    def test_get_conv_index_start_at_dilation_1_1(self):
        np.random.seed(0)
        conv2d = Conv2d(
            in_channels=2,
            out_channels=3,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=False,
            dilation=(1, 1),
            groups=1,
            bias=True,
            input_dim=5,
            trainer_type=LagrangeTrainer,
        )
        actual_conv_start_0_0 = conv2d.get_conv_index_start_at(0, 0)
        expected_conv_start_0_0 = [
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 2),
            (0, 1, 0),
            (0, 1, 1),
            (0, 1, 2),
            (0, 2, 0),
            (0, 2, 1),
            (0, 2, 2),
            (1, 0, 0),
            (1, 0, 1),
            (1, 0, 2),
            (1, 1, 0),
            (1, 1, 1),
            (1, 1, 2),
            (1, 2, 0),
            (1, 2, 1),
            (1, 2, 2),
        ]

        self.assertEqual(actual_conv_start_0_0, expected_conv_start_0_0)

        actual_conv_start_2_2 = conv2d.get_conv_index_start_at(2, 1)
        expected_conv_start_2_2 = [
            (0, 2, 1),
            (0, 2, 2),
            (0, 2, 3),
            (0, 3, 1),
            (0, 3, 2),
            (0, 3, 3),
            (0, 4, 1),
            (0, 4, 2),
            (0, 4, 3),
            (1, 2, 1),
            (1, 2, 2),
            (1, 2, 3),
            (1, 3, 1),
            (1, 3, 2),
            (1, 3, 3),
            (1, 4, 1),
            (1, 4, 2),
            (1, 4, 3),
        ]

        self.assertEqual(actual_conv_start_2_2, expected_conv_start_2_2)

    def test_get_conv_index_start_at_dilation_1_2(self):
        np.random.seed(0)
        conv2d = Conv2d(
            in_channels=2,
            out_channels=3,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=False,
            dilation=(1, 2),
            groups=1,
            bias=True,
            input_dim=5,
            trainer_type=LagrangeTrainer,
        )
        actual_conv_start_0_0 = conv2d.get_conv_index_start_at(0, 0)
        expected_conv_start_0_0 = [
            (0, 0, 0),
            (0, 0, 2),
            (0, 0, 4),
            (0, 1, 0),
            (0, 1, 2),
            (0, 1, 4),
            (0, 2, 0),
            (0, 2, 2),
            (0, 2, 4),
            (1, 0, 0),
            (1, 0, 2),
            (1, 0, 4),
            (1, 1, 0),
            (1, 1, 2),
            (1, 1, 4),
            (1, 2, 0),
            (1, 2, 2),
            (1, 2, 4),
        ]

        self.assertEqual(actual_conv_start_0_0, expected_conv_start_0_0)

        actual_conv_start_2_2 = conv2d.get_conv_index_start_at(2, 1)
        expected_conv_start_2_2 = []

        self.assertEqual(actual_conv_start_2_2, expected_conv_start_2_2)

    def test_conv2d_forward_small_image(self):
        batch = torch.rand(5, 3, 5, 5)
        conv2d = Conv2d(
            in_channels=3,
            out_channels=2,
            kernel_size=3,
            stride=(1, 1),
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            input_dim=5,
            trainer_type=LagrangeTrainer,
        )

        actual = conv2d(batch)
        self.assertEqual(actual.shape, torch.zeros(5, 2, 3, 3).shape)

    def test_conv2d_forward_large_image(self):
        batch = torch.rand(5, 3, 10, 10)
        conv2d = Conv2d(
            in_channels=3,
            out_channels=2,
            kernel_size=3,
            stride=(2, 2),
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            input_dim=10,
            trainer_type=LagrangeTrainer,
        )

        actual = conv2d(batch)
        self.assertEqual(actual.shape, torch.zeros(5, 2, 4, 4).shape)


if __name__ == "__main__":
    unittest.main()
