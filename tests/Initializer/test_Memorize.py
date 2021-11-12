import unittest
import deeplut
from deeplut.Initializer.Memorize import Memorize
import torch

from deeplut.trainer.LagrangeTrainer import LagrangeTrainer


class test_Memorize(unittest.TestCase):
    def test_weight_lookup_table_generation_k_2(self):
        trainer = LagrangeTrainer(
            tables_count=1,
            k=2,
            binary_calculations=True,
            input_expanded=True,
            device=None,
        )
        memorize = Memorize(trainer=trainer, device=None)
        memorize.generate_weight_lookup()
        self.assertEqual(
            memorize.weight_lookup_table[(-1, 1, -1, 1)], [-1, -1, 1, 1]
        )


# endregion

if __name__ == "__main__":
    unittest.main()
