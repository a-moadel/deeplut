import torch
import numpy as np
from deeplut.nn.Linear import Linear
from deeplut.trainer.LagrangeTrainer import LagrangeTrainer
from deeplut.mask.MaskExpanded import MaskExpanded
import unittest


class test_Linear(unittest.TestCase):
    def test_mask_creation(self):
        np.random.seed(0)
        linear = Linear(
            in_features=5,
            out_features=3,
            k=2,
            binary_calculations=False,
            input_expanded=True,
            trainer_type=LagrangeTrainer,
            mask_builder_type= MaskExpanded,
            device="cpu",
        )
        actual_main_achors = linear.input_mask[::2]
        total_length = 5 * 3 * 2
        expected_main_achors = torch.from_numpy(np.array(range(15)) % 5)
        self.assertEqual(total_length, linear.input_mask.shape[0])
        self.assertTrue((actual_main_achors == expected_main_achors).all())

    # to be continued
    def test_forward(self):
        np.random.seed(0)
        linear = Linear(
            in_features=5,
            out_features=3,
            k=2,
            binary_calculations=False,
            input_expanded=True,
            trainer_type=LagrangeTrainer,
            mask_builder_type= MaskExpanded,
            device="cpu",
        )
        batch_image = torch.rand(1, 5)
        linear(batch_image)


if __name__ == "__main__":
    unittest.main()
