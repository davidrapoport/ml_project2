from sklearn.feature_selection import SelectPercentile, SelectKBest, VarianceThreshold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_selection import chi2, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from classifiers.k_nearest_neighbor import KNearestNeighbor
from classifiers.naive_bayes import NaiveBayes
from vectorizers.word2vec_vectorizer import Word2VecVectorizer
import numpy as np

import pdb, random
random.seed(1234)
#lines, targets = read_input_file(quick_n_dirty=True)

count_vect = CountVectorizer(decode_error="ignore")
tfidf = TfidfVectorizer(decode_error="ignore")

vectorizers = [("count_vect", count_vect), ("tfidf", tfidf), ("wordvec", Word2VecVectorizer())]
vectorizers_params = [
        {"count_vect__ngram_range":((1,2),), "count_vect__binary":(False,), "count_vect__min_df":(2,), "count_vect__max_df":(0.3,)},
        {"tfidf__ngram_range":((1,2),), "tfidf__min_df":(2,), "tfidf__max_df":(0.3,)},
        {"wordvec__ngram_range":((1,1),(1,2)), "wordvec__min_df":(2,3), "wordvec__max_df":(0.5,), "wordvec__k":(30,50)}
        ]


kbest = SelectKBest()
percentile = SelectPercentile()
def score_func(X,y):
    return np.zeros(X.shape[1]), np.zeros(X.shape[1])
null_select = SelectKBest(score_func=score_func,k="all")

selectors = [("percentile",percentile), ("no_select", null_select)]
selectors_params = [{"percentile__percentile":(50,), "percentile__score_func":(chi2,)}, {}]

multinb = NaiveBayes(multinomial=True)
svc = SVC()
skmultinb = MultinomialNB()
dec_tree = DecisionTreeClassifier()
knn = KNeighborsClassifier()

learners = [("multinb",skmultinb), ("dec_tree", dec_tree), ("knn",knn)]
learners_params = [
        {},
        {"dec_tree__criterion":("gini","entropy")},
        {"knn__weights":("uniform","distance"), "knn__n_neighbors":(2,5,10,25)}]

