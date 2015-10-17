from sklearn.feature_selection import SelectPercentile, SelectKBest, VarianceThreshold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from classifiers.k_nearest_neighbor import KNearestNeighbor
from classifiers.naive_bayes import NaiveBayes

import pdb, random
random.seed(1234)
#lines, targets = read_input_file(quick_n_dirty=True)

count_vect = CountVectorizer(decode_error="ignore")
tfidf = TfidfVectorizer(decode_error="ignore")

vectorizers = [("count_vect", count_vect), ("tfidf", tfidf)]
vectorizers_params = [
                {"count_vect__ngram_range":((1,1), (1,2)), "count_vect__binary":(True, False)},
                        {"tfidf__ngram_range":((1,1), (1,2))}]

kbest = SelectKBest()
percentile = SelectPercentile()

selectors = [("percentile",percentile)]
selectors_params = [{"percentile__percentile":(10,25,50)}]

multinb = NaiveBayes(multinomial=True)
svc = SVC()
skmultinb = MultinomialNB()
kNN = KNearestNeighbor()

learners = [("multinomialNB",multinb),("svc", svc),("skmultinb",skmultinb)]
learners_params = [{}, {}, {}]

