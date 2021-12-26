import torch
import numpy as np
from deeplut.trainer.LagrangeTrainer import LagrangeTrainer
from deeplut.nn.utils import truth_table

import unittest


class test_LagrangeTrainer(unittest.TestCase):
    def test_forward_two_tables_one_input_k_2(self):
        lagrangeTrainer = LagrangeTrainer(
            tables_count=2,
            k=2,
            binary_calculations=False,
            input_expanded=True,
            device="cpu",
        )
        np.array([1, 2, 3, 4])
        input = torch.from_numpy(np.array([[1, 2, 3, 4]]))
        lagrangeTrainer.weight.data = torch.from_numpy(
            np.array([[5, 6, 7, 8], [9, 10, 11, 12]])
        ).float()
        output_0 = self.lagrange_calcs_k_2([5, 6, 7, 8], [1, 2])
        output_1 = self.lagrange_calcs_k_2([9, 10, 11, 12], [3, 4])
        output = lagrangeTrainer(input)
        self.assertTrue(
            (np.array([output_0, output_1]) == output.data.numpy()).all()
        )

    def test_forward_two_tables_two_input_k_2(self):
        lagrangeTrainer = LagrangeTrainer(
            tables_count=2,
            k=2,
            binary_calculations=False,
            input_expanded=True,
            device="cpu",
        )
        np.array([1, 2, 3, 4])
        input = torch.from_numpy(np.array([[1, 2, 3, 4], [1, 2, 3, 4]]))
        lagrangeTrainer.weight.data = torch.from_numpy(
            np.array([[5, 6, 7, 8], [9, 10, 11, 12]])
        ).float()
        output_0 = self.lagrange_calcs_k_2([5, 6, 7, 8], [1, 2])
        output_1 = self.lagrange_calcs_k_2([9, 10, 11, 12], [3, 4])
        output = lagrangeTrainer(input)
        self.assertTrue(
            (
                np.array([[output_0, output_1], [output_0, output_1]])
                == output.data.numpy()
            ).all()
        )

    def test_forward_two_tables_one_input_k_3(self):
        lagrangeTrainer = LagrangeTrainer(
            tables_count=2,
            k=3,
            binary_calculations=False,
            input_expanded=True,
            device="cpu",
        )
        table1_ins = [1, 2, 3]
        table2_ins = [4, 5, 6]
        input = np.array(table1_ins + table2_ins)
        inputs = torch.from_numpy(np.array([input]))

        tabel1_weights = [5, 6, 7, 8, 9, 10, 11, 12]
        tabel2_weights = [13, 14, 15, 16, 17, 18, 19, 20]

        lagrangeTrainer.weight.data = torch.from_numpy(
            np.array([tabel1_weights, tabel2_weights])
        ).float()
        output_0 = self.lagrange_calcs_k_3(tabel1_weights, table1_ins)
        output_1 = self.lagrange_calcs_k_3(tabel2_weights, table2_ins)
        output = lagrangeTrainer(inputs)
        self.assertTrue(
            (np.array([output_0, output_1]) == output.data.numpy()).all()
        )

    def test_random_large_test_seeded_k_2(self):
        self.random_testing_seeded(k=2, iterations_count=20)

    def test_random_large_test_seeded_k_3(self):
        self.random_testing_seeded(k=3, iterations_count=20)

    def test_forward_two_tables_one_input_k_2_input_not_expanded(self):
        lagrangeTrainer = LagrangeTrainer(
            tables_count=2,
            k=2,
            binary_calculations=False,
            input_expanded=False,
            device="cpu",
        )
        np.array([1, 2, 3, 4])
        input = torch.from_numpy(np.array([[1, 2, 3, 4]]))
        lagrangeTrainer.weight.data = torch.from_numpy(
            np.array([[5, 6, 7, 8], [9, 10, 11, 12]])
        ).float()
        output_0 = self.lagrange_calcs_k_2_not_expanded([5, 6, 7, 8], [1, 2])
        output_1 = self.lagrange_calcs_k_2_not_expanded(
            [9, 10, 11, 12], [3, 4]
        )
        output = lagrangeTrainer(input)
        self.assertTrue(
            (np.array([output_0, output_1]) == output.data.numpy()).all()
        )

    def test_generate_weight_lookup_k_2(self):
        trainer = LagrangeTrainer(1, 2, True, True, None)
        weight_lookup_table = trainer.generate_weight_lookup()
        inputs = truth_table.generate_truth_table(
            k=2, tables_count=1, device=None)
        for expected_output, weights in weight_lookup_table.items():
            trainer.weight.data = torch.tensor(
                weights, dtype=torch.float32, requires_grad=True)
            actual_output = trainer(inputs.T).detach(
            ).cpu().int().flatten().numpy().tolist()
            self.assertEqual(list(expected_output), actual_output)

    def random_testing_seeded(self, k, iterations_count):
        maximum_batch_size = 100
        maximum_table_count = 200
        np.random.seed(0)
        for i in range(iterations_count):
            batch_size = np.random.randint(maximum_batch_size) + 10
            table_count = np.random.randint(maximum_table_count) + 10
            input_size = table_count * k
            inputs_list = []
            for id in range(batch_size):
                inputs_list.append(
                    list(
                        np.random.rand(
                            input_size,
                        )
                    )
                )
            inputs = torch.from_numpy(np.array(inputs_list))

            lagrangeTrainer = LagrangeTrainer(
                tables_count=table_count,
                k=k,
                binary_calculations=False,
                input_expanded=True,
                device="cpu",
            )

            weights_list = []
            for id in range(table_count):
                weights_list.append(
                    list(lagrangeTrainer.weight[id].detach().numpy())
                )

            outputs = []

            for id in range(batch_size):
                result = []
                for idw in range(table_count):
                    start = idw * k
                    end = (idw + 1) * k
                    result.append(
                        self.lagrange_calcs(
                            weights_list[idw], inputs[id, start:end], k
                        )
                    )
                outputs.append(result)

            actual_output = lagrangeTrainer(inputs)
            expected_output = np.array(outputs)
            allowed_error = 1e-5
            condition = (
                actual_output.detach().numpy() - expected_output
            ) < allowed_error
            self.assertTrue(condition.all())

    def lagrange_calcs(self, weights, inputs, k):
        if k == 2:
            return self.lagrange_calcs_k_2(weights, inputs)
        elif k == 3:
            return self.lagrange_calcs_k_3(weights, inputs)
        else:
            raise Exception("Invalid k")

    def lagrange_calcs_k_2(self, weights, inputs):

        return (
              weights[0] * (1 - inputs[0]) * (1 - inputs[1])
            + weights[1] * (1 - inputs[0]) * (1 + inputs[1])
            + weights[2] * (1 + inputs[0]) * (1 - inputs[1])
            + weights[3] * (1 + inputs[0]) * (1 + inputs[1])
        )

    def lagrange_calcs_k_3(self, weights, inputs):

        return (
              weights[0] * (1 -inputs[0]) * (1 -inputs[1]) * (1 - inputs[2])
            + weights[1] * (1- inputs[0]) * (1 - inputs[1]) * (1 + inputs[2])
            + weights[2] * (1- inputs[0]) * (1 + inputs[1]) * (1 - inputs[2])
            + weights[3] * (1- inputs[0]) * (1 + inputs[1]) * (1 + inputs[2])
            + weights[4] * (1 + inputs[0]) * (1 - inputs[1]) * (1 - inputs[2])
            + weights[5] * (1 + inputs[0]) * (1 - inputs[1]) * (1 + inputs[2])
            + weights[6] * (1 + inputs[0]) * (1 + inputs[1]) * (1 - inputs[2])
            + weights[7] * (1 + inputs[0]) * (1 + inputs[1]) * (1 + inputs[2])
        )

    def lagrange_calcs_k_2_not_expanded(self, weights, inputs):

        return weights[0] * (1 - inputs[0])

    def lagrange_calcs_k_3_not_expanded(self, weights, inputs):

        return weights[0] * (1 - inputs[0])


if __name__ == "__main__":
    unittest.main()
