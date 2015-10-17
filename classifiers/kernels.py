__author__ = 'Charlie'

"""
Collection of distance functions to be used to measure the likeness of two vectors.
Each function takes numpy 1d numpy arrays as parameters and returns a likeness score.

The functions must return larger values for points that are less alike.
"""

import numpy as np

def euclidian_distance(source, target):
    """
    :param source: 1d numpy array
    :param target: 1d numpy array
    :return: Euclidian distance between the two vectors
    """
    return np.linalg.norm(source - target)

def cosine_distance(source, target):
    """
    :param source: 1d numpy array
    :param target: 1d numpy array
    :return: 1 minus the cosine similarity between to vectors
    :see: https://en.wikipedia.org/wiki/Cosine_similarity
    """
    def cosine_similarity(s, t):
        return np.inner(s, t) / (np.linalg.norm(s) * np.linalg.norm(t))

    return 1 - cosine_similarity(source, target)