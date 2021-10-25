import unittest
import torch
import numpy as np
from deeplut.nn.utils import truth_table


class TestTruthTableGeneration(unittest.TestCase):
    def test_basic_scenario_k_2(self):
        actual = truth_table.generate_truth_table(2, 3, "cpu")

        expected = torch.from_numpy(
            np.array(
                [
                    [-1, -1, 1, 1],
                    [-1, 1, -1, 1],
                    [-1, -1, 1, 1],
                    [-1, 1, -1, 1],
                    [-1, -1, 1, 1],
                    [-1, 1, -1, 1],
                ]
            )
        )
        equal_result = torch.eq(actual, expected)
        self.assertTrue(torch.all(equal_result))

    def test_basic_scenario_k_3(self):
        actual = truth_table.generate_truth_table(3, 2, "cpu")

        expected = torch.from_numpy(
            np.array(
                [
                    [-1, -1, -1, -1, 1, 1, 1, 1],
                    [-1, -1, 1, 1, -1, -1, 1, 1],
                    [-1, 1, -1, 1, -1, 1, -1, 1],
                    [-1, -1, -1, -1, 1, 1, 1, 1],
                    [-1, -1, 1, 1, -1, -1, 1, 1],
                    [-1, 1, -1, 1, -1, 1, -1, 1],
                ]
            )
        )
        equal_result = torch.eq(actual, expected)
        self.assertTrue(torch.all(equal_result))

    def test_odd_scenario_k_0(self):
        actual = truth_table.generate_truth_table(0, 3, "cpu")

        expected = torch.from_numpy(np.array([]))
        equal_result = torch.eq(actual, expected)
        self.assertTrue(torch.all(equal_result))

    def test_odd_scenario_tables_count_0(self):
        with self.assertRaises(RuntimeError):
            truth_table.generate_truth_table(2, 0, "cpu")

    def test_missing_argument_tables_count(self):
        with self.assertRaises(TypeError):
            _ = truth_table.generate_truth_table(0, device="cpu")

    def test_missing_argument_device(self):
        with self.assertRaises(TypeError):
            truth_table.generate_truth_table(0, 1)

    def test_missing_argument_k(self):
        with self.assertRaises(TypeError):
            _ = truth_table.generate_truth_table(tables_count=1, device="cpu")


class TestReduceTruthTable(unittest.TestCase):
    def test_reduce_2d_table_k_2_all_ones(self):
        table = torch.from_numpy(
            np.array(
                [
                    [-1, -1, 1, 1],
                    [-1, 1, -1, 1],
                    [-1, -1, 1, 1],
                    [-1, 1, -1, 1],
                    [-1, -1, 1, 1],
                    [-1, 1, -1, 1],
                ]
            )
        )
        actual = truth_table.reduce_truth_table(2, table, "cpu")
        expected = torch.from_numpy(
            np.array([[1, -1, -1, 1], [1, -1, -1, 1], [1, -1, -1, 1]])
        )
        equal_result = torch.eq(actual, expected)
        self.assertTrue(torch.all(equal_result))

    def test_reduce_2d_table_k_2_arbitrary_numbers(self):
        table = torch.from_numpy(
            np.array(
                [
                    [-1, -2, 3, 4],
                    [-5, 6, -7, 8],
                    [-9, -10, 11, 12],
                    [-13, 14, -15, 16],
                    [-17, -18, 19, 20],
                    [-21, 22, -23, 24],
                ]
            )
        )
        actual = truth_table.reduce_truth_table(2, table, "cpu")
        expected = torch.from_numpy(
            np.array(
                [
                    [-1 * -5, -2 * 6, 3 * -7, 4 * 8],
                    [-9 * -13, -10 * 14, 11 * -15, 12 * 16],
                    [-17 * -21, -18 * 22, 19 * -23, 20 * 24],
                ]
            )
        )
        equal_result = torch.eq(actual, expected)
        self.assertTrue(torch.all(equal_result))


if __name__ == "__main__":
    unittest.main()
