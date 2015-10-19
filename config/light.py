from sklearn.feature_selection import SelectPercentile, SelectKBest, VarianceThreshold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_selection import chi2, f_classif

from classifiers.naive_bayes import NaiveBayes

import pdb, random
random.seed(1234)
#lines, targets = read_input_file(quick_n_dirty=True)

count_vect = CountVectorizer(decode_error="ignore")
tfidf = TfidfVectorizer(decode_error="ignore")

vectorizers = [("count_vect", count_vect), ("tfidf", tfidf)]
vectorizers_params = [
        {"count_vect__ngram_range":((1,1),), "count_vect__binary":(True,), "count_vect__min_df":(2,)},
        {"tfidf__ngram_range":((1,1),), "tfidf__min_df":(2,)}
        ]


kbest = SelectKBest()
percentile = SelectPercentile()

selectors = [("percentile",percentile)]
selectors_params = [{"percentile__percentile":(10,), "percentile__score_func":(f_classif,)}]

multinb = NaiveBayes(multinomial=True)
svc = SVC()

learners = [("multinomialNB",multinb),("svc", svc)]
learners_params = [{}, {}, {}]

