__author__ = 'Charlie'

import numpy as np
from sklearn import cross_validation


def evaluate_classifier(X, Y, learner):
    """
    :param X: feature set
    :param Y: classification set
    :param learner: learner class
    :return: list of classification percentages
    """
    N, _ = X.shape
    kf = cross_validation.KFold(n=N, n_folds=10, shuffle=True)
    results = []

    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        l = learner()
        l.train(X_train, Y_train)
        predictions = l.predict(X_test)
        percent_correct = np.mean(predictions != Y_test)
        results.append(percent_correct)

    return np.mean(results), np.std(results), results


