__author__ = 'Charlie'


import classifiers.k_nearest_neighbor as knn
import generate_dataset
import learner_engine


X_unigram, Y_unigram, _ = generate_dataset.generate_unigram_bernoulli(False)
mean, std, res = learner_engine.evaluate_classifier(X_unigram, Y_unigram, knn.KNearestNeighbor)

print ("Mean classification percentage = {}".format(mean))
print ("Standard deviation of classification percentages = {}".format(std))