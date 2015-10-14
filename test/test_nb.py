from classifiers.naive_bayes import NaiveBayes
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
import generate_dataset as gd
import pdb

X,y, _ = gd.generate_test1()

g = GaussianNB()
n = NaiveBayes(continuous=True)
#g.fit(X[:80,:].toarray(),y[:80])
#n.fit(X[:80,:],y[:80])
#print n.score(X[80:], y[80:])
#print g.score(X[80:].toarray(), y[80:])

X,y, c = gd.generate_unigram_multinomial(False)
N,M = X.shape
train = 8.0*N/10.0
g = MultinomialNB()
n = NaiveBayes(multinomial=True)
g.fit(X[:train],y[:train])
n.fit(X[:train],y[:train])
print n.score(X[train:],y[train:])
print g.score(X[train:],y[train:])
