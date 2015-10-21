__author__ = 'Charlie'

import random

from sklearn.feature_selection import SelectPercentile, SelectKBest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.feature_selection import chi2

random.seed(1234)
#lines, targets = read_input_file(quick_n_dirty=True)

count_vect = CountVectorizer(decode_error="ignore")
tfidf = TfidfVectorizer(decode_error="ignore")

vectorizers = [
    ("count_vect", count_vect),
    ("tfidf", tfidf)
]
vectorizers_params = [
    {
        "count_vect__ngram_range": ((1, 2),)
    },
    {
        "tfidf__ngram_range": ((1, 2),)
    }
]


kbest = SelectKBest()
percentile = SelectPercentile()

selectors = [
    ("percentile", percentile)
]
selectors_params = [
    {
        "percentile__percentile": (5,),
        "percentile__score_func": (chi2,)
    }
]

svc = SVC()

learners = [
    ("svc", svc)
]

learners_params = [
    {
        "svc__kernel": ("linear",),
        "svc__C": (2.5,)
    }
]

