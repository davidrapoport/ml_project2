from sklearn.feature_selection import SelectPercentile, SelectKBest, VarianceThreshold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2, f_classif

from classifiers.naive_bayes import NaiveBayes

import pdb, random
random.seed(1234)
#lines, targets = read_input_file(quick_n_dirty=True)

count_vect = CountVectorizer(decode_error="ignore")
tfidf = TfidfVectorizer(decode_error="ignore")

vectorizers = [("count_vect", count_vect)]
vectorizers_params = [
        {"count_vect__ngram_range":((1,2),), "count_vect__binary":(False,), "count_vect__min_df":(2,), "count_vect__max_df":(1.0,)},
        ]


kbest = SelectKBest()
percentile = SelectPercentile()

selectors = [("percentile",percentile)]
selectors_params = [{"percentile__percentile":(50,), "percentile__score_func":(chi2,)}]

multinb = NaiveBayes(multinomial=True, cnb=True)

learners = [("cnb",multinb)] 
learners_params = [
        {},
        ]
