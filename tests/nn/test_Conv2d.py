import torch
import numpy as np
from deeplut.nn.Conv2d import Conv2d
from deeplut.trainer.LagrangeTrainer import LagrangeTrainer
from deeplut.mask.MaskExpanded import MaskExpanded
import unittest


class test_Conv2d(unittest.TestCase):
    def test_correct_conv2d_calculations(self):
        batch = torch.rand(5, 3, 5, 5)
        torch_conv = torch.nn.Conv2d(3, 2, 3, bias=False)
        torch_output = torch_conv(batch)
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
        conv2d.trainer.weight.data = torch.zeros_like(
            conv2d.trainer.weight.data
        )
        conv2d.trainer.weight[:, 0].data.copy_(torch_conv.weight.data.view(-1))
        conv2d.trainer.set_binary_calculations(False)
        conv2d.trainer.set_input_expanded(False)
        dlut_output = conv2d(batch)
        self.assertEqual(dlut_output.shape, torch_output.shape)
        self.assertTrue((torch.abs(dlut_output - torch_output) < 1e-6).all())

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
