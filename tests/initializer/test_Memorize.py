import random
import unittest
import numpy as np
import torch
from deeplut.initializer.Memorize import Memorize
from deeplut.trainer.LagrangeTrainer import LagrangeTrainer
from deeplut.nn.utils import truth_table


class test_Memorize(unittest.TestCase):
    def test_input_to_id_one_by_one(self):
        trainer = LagrangeTrainer(1, 2, 3, True, None)
        memorize_initializer = Memorize(
            trainer.tables_count, trainer.k, trainer.kk, None
        )
        inputs = truth_table.generate_truth_table(
            k=2, tables_count=1, device=None
        ).T
        for expected_id in range(4):
            actual_id = memorize_initializer._input_to_id(
                inputs[expected_id].view(1, 2)
            )
            self.assertEqual(expected_id, int(actual_id.numpy()))

    def test_input_to_id_vectorized(self):
        trainer = LagrangeTrainer(1, 2, 3, True, None)
        memorize_initializer = Memorize(
            trainer.tables_count, trainer.k, trainer.kk, None
        )
        inputs = truth_table.generate_truth_table(
            k=2, tables_count=1, device=None
        ).T
        actual_id = memorize_initializer._input_to_id(inputs)
        self.assertEqual([0, 1, 2, 3], list(actual_id.numpy()))

    def test_input_to_id_vectorized_mutiple_tables(self):
        trainer = LagrangeTrainer(1, 2, 3, True, None)
        memorize_initializer = Memorize(
            trainer.tables_count, trainer.k, trainer.kk, None
        )
        inputs = truth_table.generate_truth_table(
            k=2, tables_count=1, device=None
        ).T.repeat(2, 2)
        actual_id = memorize_initializer._input_to_id(inputs)
        expected_id = np.array(
            [[0, 0], [1, 1], [2, 2], [3, 3], [0, 0], [1, 1], [2, 2], [3, 3]]
        )
        self.assertTrue((actual_id.numpy() == expected_id).all())

    def test_update_tables_weights(self):
        trainer = LagrangeTrainer(1, 2, 3, True, None)
        memorize_initializer = Memorize(
            trainer.tables_count, trainer.k, trainer.kk, None
        )
        inputs = truth_table.generate_truth_table(
            k=2, tables_count=1, device=None
        ).T
        targets = torch.tensor([0, 0, 1, 0]).view(4, 1)
        memorize_initializer.clear()
        memorize_initializer.update_counter(inputs, targets)
        trainer.weight.data = memorize_initializer.update_luts_weights()

        actual_output = trainer(inputs).detach().cpu().flatten().numpy()
        expected_output = np.array([-1, -1, 1, -1])

        self.assertTrue((expected_output == actual_output).all())

    def test_update_tables_weights_can_not_decide_senario(self):
        trainer = LagrangeTrainer(1, 2, 3, True, None)
        memorize_initializer = Memorize(
            trainer.tables_count, trainer.k, trainer.kk, None
        )
        inputs = torch.tensor([[-1, 1], [-1, 1], [1, 1], [1, 1]])
        targets = torch.tensor([0, 1, 0, 1]).view(4, 1)
        memorize_initializer.clear()
        memorize_initializer.update_counter(inputs, targets)
        random.seed(10)
        trainer.weight.data = memorize_initializer.update_luts_weights()

        actual_output = trainer(inputs).detach().cpu().flatten().numpy()
        expected_output = np.array([1, 1, 1, 1])

        self.assertTrue((expected_output == actual_output).all())


if __name__ == "__main__":
    unittest.main()
