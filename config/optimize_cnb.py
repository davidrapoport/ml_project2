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

import pdb, random
random.seed(1234)
#lines, targets = read_input_file(quick_n_dirty=True)

count_vect = CountVectorizer(decode_error="ignore")
tfidf = TfidfVectorizer(decode_error="ignore")

vectorizers = [("count_vect", count_vect), ("tfidf", tfidf)]
vectorizers_params = [
        {"count_vect__ngram_range":((1,2),(1,1)), "count_vect__binary":(False,), "count_vect__min_df":(2,3,), "count_vect__max_df":(0.5,1.0,0.75)},
        {"tfidf__ngram_range":((1,2),), "tfidf__min_df":(2,3,), "tfidf__max_df":(0.3,0.5,1.0)}
        ]


kbest = SelectKBest()
percentile = SelectPercentile()

selectors = [("percentile",percentile)]
selectors_params = [{"percentile__percentile":(10,50,100), "percentile__score_func":(chi2,)}]

mnb = NaiveBayes(multinomial=True)
cnb = NaiveBayes(multinomial=True, cnb=True)
wcnb = NaiveBayes(multinomial=True, wcnb=True)

learners = [("mnb",mnb), ("cnb", cnb), ("wcnb",wcnb)]
learners_params = [
        {},
        {"cnb__alpha":(1.0,2.0,5.0)},
        {}
        ]
