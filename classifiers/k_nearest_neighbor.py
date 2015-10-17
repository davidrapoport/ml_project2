__author__ = 'Charlie'

import numpy as np
import heapq
from classifiers.kernels import cosine_distance


class KNearestNeighbor(object):
    """
    Bias/Variance Tradeoff

    Bias tends to be high when k is high and low when k is low.
    Variance tends to be high when k is low and low when k is high
    """

    def __init__(self, k=1, distance=cosine_distance):
        self.k = k
        self.distance = distance

    def get_params(self, *args, **kwargs):
        return {
            "k": self.k,
        }

    def fit(self, features, targets):
        """
        No 'training' is needed with k nearest neighbor.
        Classification is performed at prediction time.

        :param features: 2d numpy matrix of feature vectors
        :param targets: 1d numpy matrix of corresponding classifications
        :return: void
        """
        self.features = features
        self.targets = targets
        self.N, self.M = features.shape

    def _classify(self, vector):
        """
        :param vector: 1d numpy array, feature vector
        :return: int, representing the feature vector's classification
        """
        print vector

        neighbors = []
        for index in range(self.N):
            neighbor_class = self.targets[index]
            neighbor_distance = self.distance(vector, self.features[index])
            heapq.heappush(neighbors, (neighbor_distance, neighbor_class))

        k_nearest = heapq.nlargest(self.k, neighbors)  # Take top k values off heap as the k nearest neighbors
        k_nearest_classes = map(lambda tup: tup[1], k_nearest)  # Take only classes of the k nearest neighbors
        most_common_class = max(set(k_nearest_classes), key=k_nearest_classes.count)
        return most_common_class

    def predict(self, observed):
        """
        :param observed: 2d numpy matrix of observed test points
        :return: 1d numpy matrix of predicted classifications
        """
        #predictions = np.array(self.N)
        return np.apply_along_axis(self._classify, axis=1, arr=observed)

    def score(self, X, Y):
        """
        :param X:
        :param Y:
        :return: classification success
        """
        yp = self.predict(X).ravel()
        return float((yp == Y).sum())/Y.shape[0]


