import unittest
import deeplut
import torch
import numpy as np
from deeplut.nn.utils import MaskBuilder


class test_MaskBuilder(unittest.TestCase):

    def test_mask_builder_normal_scenario(self):
        np.random.seed(0)
        table_input_selections = [((0, 0), [(0, 1), (0, 2)]), ((
            0, 1), [(0, 0), (0, 2)]), ((0, 2), [(0, 0), (0, 1)])]
        maskBuilder = MaskBuilder(3, 2, table_input_selections, True)
        result = maskBuilder.build()
        tmp = 0

if __name__ == '__main__':
    unittest.main()
