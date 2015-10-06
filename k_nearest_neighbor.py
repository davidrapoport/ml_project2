__author__ = 'Charlie'

import numpy as np
import heapq

class KNearestNeighbor(object):

    def __init__(self, k=1):
        self.k = k

    def train(self, features, targets):
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
        neighbors = []
        for index, row in enumerate(self.features.tolist()):
            neighbor_class = self.targets[index]
            neighbor_distance = np.linalg.norm(vector - np.array(row))
            heapq.heappush(neighbors, (neighbor_distance, neighbor_class))

        k_nearest = heapq.nlargest(self.k, neighbors) #take top k values off heap as the k nearest neighbors
        k_nearest_classes = map(lambda tup: tup[1], k_nearest) #take only classes of the k nearest neighbors
        most_common_class = max(set(k_nearest_classes), k_nearest_classes.count)
        return most_common_class

    def predict(self, observed):
        """
        :param observed: 2d numpy matrix of observed test points
        :return: 1d numpy matrix of predicted classifications
        """
        return np.apply_along_axis(self._classify, axis=1, arr=observed)

