from classifiers.naive_bayes import NaiveBayes
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import generate_dataset as gd
import pdb, unittest
import numpy as np
import random
import math
random.seed(1234)

lines, targets = gd.read_input_file(quick_n_dirty=True)
N = len(lines)
lines_and_targets = zip(lines,targets)
random.shuffle(lines_and_targets)
lines, targets = zip(*lines_and_targets)
targets = np.vstack(targets)

train_lines = lines[:8*N/10]
train_targets = targets[:8*N/10].ravel()

test_lines = lines[8*N/10:]
test_targets = targets[8*N/10:].ravel()

count_vect = CountVectorizer()
X_count = count_vect.fit_transform(train_lines)
X_count_test = count_vect.transform(test_lines)

tfidf_vect = TfidfVectorizer()
X_tfidf = tfidf_vect.fit_transform(train_lines)
X_tfidf_test = tfidf_vect.transform(test_lines)

class NBTest(unittest.TestCase):

    def setUp(self):
        self.mnb = NaiveBayes(multinomial=True)
        self.skmnb = MultinomialNB()
        self.bnb = NaiveBayes(bernoulli=True)
        self.skbnb = BernoulliNB()
        self.cnb = NaiveBayes(multinomial=True, cnb=True)
        self.wcnb = NaiveBayes(multinomial=True, wcnb=True)

    def test_count_vectorized(self):
        self.mnb.fit(X_count, train_targets)
        self.skmnb.fit(X_count, train_targets)
        self.assertEqual(self.mnb.score(X_count_test,test_targets),self.skmnb.score(X_count_test,test_targets))

    def test_tfidf_vectorized(self):
        self.mnb.fit(X_tfidf, train_targets)
        self.skmnb.fit(X_tfidf, train_targets)
        self.assertEqual(self.mnb.score(X_tfidf_test, test_targets), self.skmnb.score(X_tfidf_test, test_targets))

    def test_cnb(self):
        self.cnb.fit(X_count, train_targets)
        self.mnb.fit(X_count, train_targets)
        cnb_score = self.cnb.score(X_count_test, test_targets)
        mnb_score = self.mnb.score(X_count_test, test_targets)
        print "CNB: {},   MNB: {}".format(cnb_score, mnb_score)
        assert (cnb_score - mnb_score) > -0.1  

    def test_wcnb(self):
        self.wcnb.fit(X_count, train_targets)
        self.mnb.fit(X_count, train_targets)
        wcnb_score = self.wcnb.score(X_count_test, test_targets)
        mnb_score = self.mnb.score(X_count_test, test_targets)
        print "WCNB: {},   MNB: {}".format(wcnb_score, mnb_score)
        assert (wcnb_score - mnb_score) > -0.5  
