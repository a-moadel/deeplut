import torch
import numpy as np
from deeplut.nn.Conv2d import Conv2d
from deeplut.trainer.LagrangeTrainer import LagrangeTrainer
from deeplut.mask.MaskExpanded import MaskExpanded
import unittest


class test_Conv2d(unittest.TestCase):
    def test_conv2d_forward_small_image(self):
        batch = torch.rand(5, 3, 5, 5)
        conv2d = Conv2d(
            in_channels=3,
            out_channels=2,
            kernel_size=3,
            trainer_type=LagrangeTrainer,
            mask_builder_type=MaskExpanded,
            stride=(1, 1),
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            input_dim=5,
        )

        actual = conv2d(batch)
        self.assertEqual(actual.shape, torch.zeros(5, 2, 3, 3).shape)

    def test_conv2d_forward_large_image(self):
        batch = torch.rand(5, 3, 10, 10)
        conv2d = Conv2d(
            in_channels=3,
            out_channels=2,
            kernel_size=3,
            trainer_type=LagrangeTrainer,
            mask_builder_type=MaskExpanded,
            stride=(2, 2),
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            input_dim=10,
        )

        actual = conv2d(batch)
        self.assertEqual(actual.shape, torch.zeros(5, 2, 4, 4).shape)


if __name__ == "__main__":
    unittest.main()
