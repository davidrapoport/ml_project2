__author__ = 'Charlie'

import classifiers.k_nearest_neighbor as kNN
import unittest
import generate_dataset


class TestClassifiers(unittest.TestCase):

    def test_k_nearest_neighbor_no_runtime_errors(self):
        k = kNN.KNearestNeighbor()
        X, Y, _ = generate_dataset.generate_test1()

        k.train(X, Y)
        predictions = k.predict(X)

        self.assertEqual(predictions.shape, X.shape)


if __name__ == '__main__':
    unittest.main()
