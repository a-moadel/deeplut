import unittest
import numpy as np
from deeplut.mask.MaskMinimal import MaskMinimal


class test_MaskMinimal(unittest.TestCase):
    
    def test_mask_builder_normal_scenario(self):
        np.random.seed(0)
        table_input_selections = [
            ((0, 0), [(0, 1), (0, 2)]),
            ((0, 1), [(0, 0), (0, 2)]),
            ((0, 2), [(0, 0), (0, 1)]),
        ]
        maskBuilder = MaskMinimal(2, table_input_selections, True)
        actual_mask = maskBuilder.build()
        expected_mask = np.array(
            [[0, 0], [0, 1], [0, 2], [0, 0]]
        )
        self.assertTrue((actual_mask == expected_mask).all())

if __name__ == "__main__":
    unittest.main()
