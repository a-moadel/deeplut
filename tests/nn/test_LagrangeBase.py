import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Final
import numpy as np
from deeplut.nn import LagrangeBase
import unittest

class test_LagrangeBase(unittest.TestCase):
    
    def test_forward_two_tables_one_input(self):
        lagrangeBase = LagrangeBase(tables_count=2, k=2, binary_calculations=False, device='cpu')
        image = np.array([1,2,3,4])
        input = torch.from_numpy(np.array([[1,2,3,4]]))
        lagrangeBase.weight.data = torch.from_numpy(np.array([[5,6,7,8],[9,10,11,12]])).float()
        output_0 = self.lagrange_calcs_k_2([5,6,7,8],[1,2])
        output_1 = self.lagrange_calcs_k_2([9,10,11,12],[3,4])
        output = lagrangeBase(input)
        self.assertTrue((np.array([output_0,output_1]) == output.data.numpy()).all())

    def test_forward_two_tables_two_input(self):
        lagrangeBase = LagrangeBase(tables_count=2, k=2, binary_calculations=False, device='cpu')
        image = np.array([1,2,3,4])
        input = torch.from_numpy(np.array([[1,2,3,4],[1,2,3,4]]))
        lagrangeBase.weight.data = torch.from_numpy(np.array([[5,6,7,8],[9,10,11,12]])).float()
        output_0 = self.lagrange_calcs_k_2([5,6,7,8],[1,2])
        output_1 = self.lagrange_calcs_k_2([9,10,11,12],[3,4])
        output = lagrangeBase(input)
        self.assertTrue((np.array([[output_0,output_1],[output_0,output_1]]) == output.data.numpy()).all())

    def lagrange_calcs_k_2(self, weights,inputs):
        
        return weights[0]*(inputs[0]-1)*(inputs[1]-1) + \
               weights[1]*(inputs[0]-1)*(inputs[1]+1) + \
               weights[2]*(inputs[0]+1)*(inputs[1]-1) + \
               weights[3]*(inputs[0]+1)*(inputs[1]+1)

if __name__ == '__main__':
    unittest.main()