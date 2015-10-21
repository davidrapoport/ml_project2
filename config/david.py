from sklearn.feature_selection import SelectPercentile, SelectKBest, VarianceThreshold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import chi2, f_classif

from classifiers.k_nearest_neighbor import KNearestNeighbor
from classifiers.naive_bayes import NaiveBayes
from vectorizers.word2vec_vectorizer import Word2VecVectorizer
import numpy as np

import pdb, random
random.seed(1234)
#lines, targets = read_input_file(quick_n_dirty=True)

count_vect = CountVectorizer(decode_error="ignore")

vectorizers = [("count_vect", count_vect), ("wordvec", Word2VecVectorizer(filter_type="top_k"))]
vectorizers_params = [
        {"count_vect__ngram_range":((1,2),), "count_vect__binary":(False,), "count_vect__min_df":(2,), "count_vect__max_df":(0.3,)},
        {"wordvec__ngram_range":((1,1),), "wordvec__min_df":(2,4), "wordvec__k":(100,200)}
        ]


kbest = SelectKBest()
percentile = SelectPercentile()
def score_func(X,y):
    return np.zeros(X.shape[1]), np.zeros(X.shape[1])
null_select = SelectKBest(score_func=score_func,k="all")

selectors = [("percentile",percentile), ("no_select", null_select)]
selectors_params = [{"percentile__percentile":(50,), "percentile__score_func":(chi2,)}, {}]

multinb = NaiveBayes(multinomial=True, cnb=True)
svc = SVC(cache_size=1000)
linsvc = LinearSVC()

learners = [("cnb",multinb),("svc",svc),("linsvc", linsvc)]
learners_params = [
        {},
        {"svc__kernel":("poly","rbf","sigmoid"), "svc__C":(0.5,1.0,2.0)},
        {"linsvc__c":(0.5,1.0,2.0)}
        ]
