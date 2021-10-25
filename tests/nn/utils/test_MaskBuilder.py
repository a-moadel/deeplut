import unittest
import numpy as np
from deeplut.nn.utils.MaskBuilder import MaskBuilder


class test_MaskBuilder(unittest.TestCase):
    def test_mask_builder_normal_scenario(self):
        np.random.seed(0)
        table_input_selections = [
            ((0, 0), [(0, 1), (0, 2)]),
            ((0, 1), [(0, 0), (0, 2)]),
            ((0, 2), [(0, 0), (0, 1)]),
        ]
        maskBuilder = MaskBuilder(2, table_input_selections, True)
        actual_mask = maskBuilder.build_expanded()
        expected_mask = np.array(
            [[0, 0], [0, 1], [0, 1], [0, 2], [0, 2], [0, 1]]
        )
        self.assertTrue((actual_mask == expected_mask).all())


if __name__ == "__main__":
    unittest.main()
