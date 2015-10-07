__author__ = 'Charlie'

import numpy as np


class DecisionTree(object):

    def __init__(self):
        """
        :return:
        """
        self.root = None

    def train(self, features, targets):
        """
        :param features:
        :param targets:
        :return: void
        """
        self.root = self._construct_tree(features, targets)

    def _construct_tree(self, features, targets):
        """
        :param features:
        :param targets:
        :return:
        """
        if self._same_class(targets):
            return LeafNode(targets[0])

        else:
            children = []
            feature_groupings, target_groupings, classifier = self._split(features, targets)
            for feature_group, target_group in zip(feature_groupings, target_groupings):
                child = self._construct_tree(feature_group, target_group)
                children.append(child)
            return DTNode(children, classifier)

    def _split(self, features, targets):
        """
        TODO

        :param features:
        :param targets:
        :return: list of features, list of groups, classifier
        """
        return [], [], None

    def _same_class(self, targets):
        """
        TODO

        :param targets:
        :return:
        """
        return True

    def predict(self, observed):
        """
        TODO

        :param observed:
        :return:
        """
        return np.apply_along_axis(self._classify, 0, observed)

    def _classify(self, instance):
        """
        :param instance:
        :return: classification of instance by delegating to the root DTNode classifier
        """
        return self.root.classify(instance)


class DTNode(object):

    def __init__(self, children, classifier):
        self.children = children
        self.classifier = classifier

    def classify(self, instance):
        return None


class LeafNode(object):

    def __init__(self, classification):
        self.classification = classification

    def classify(self, instance):
        return self.classification
