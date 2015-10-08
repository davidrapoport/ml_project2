__author__ = 'Charlie'

import classifiers.k_nearest_neighbor
import unittest
import generate_dataset

class TestClassifiers(unittest.TestCase):

    def test_k_nearest_neighbor_no_runtime_errors(self):
        knn = classifiers.k_nearest_neighbor.KNearestNeighbor()
        X, Y, _ = generate_dataset.generate_test1()

        knn.train(X, Y)
        predictions = knn.predict(X)

        self.assertEqual(predictions.shape, X.shape)


if __name__ == '__main__':
    unittest.main()
