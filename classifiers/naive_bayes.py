from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import pdb, math
import numpy as np

class NaiveBayes(object):

    def __init__(self, l=1.0, alpha=2.0, multinomial=False, continuous=False, bernoulli=True):
        self.l = l 
        self.alpha = alpha
        self.multinomial = multinomial
        self.continuous = continuous
        self.bernoulli = bernoulli and (not multinomial and not continuous)
        
    def fit(self, features, targets):
        self.classes = list(np.unique(targets))
        self.num_classes = len(self.classes) 
        self.classes_array = np.zeros((self.num_classes,1))
        self.classes_array[:,0]=self.classes
        N,M = features.shape
        self.M = M

        # Compute p(yi) for all classes
        self.py = np.zeros((self.num_classes,1)) #prob(yi)
        self.num_y = np.zeros((self.num_classes,1))
        for cnt,y in enumerate(self.classes):
            self.num_y[cnt] = (targets==y).sum()
            self.py[cnt] = self.num_y[cnt]/N
        
        if self.bernoulli:
            self._train_bernoulli(features, targets)
        elif self.multinomial:
            self._train_multinomial(features, targets)
        else:
            self._train_continuous(features, targets)

    def _train_bernoulli(self, features, targets):
        self.x_given_y = np.zeros((self.M, self.num_classes))
        self.x_and_y_count = np.zeros((self.M, self.num_classes))
        feature_transpose = features.transpose()
        targets_transpose = targets.transpose()[0]
        for cnt, y in enumerate(self.classes):
            X_in_class = features[targets_transpose==y]
            self.x_given_y[:, cnt] = (X_in_class.sum(0) + self.l)/(self.num_y[cnt] + self.alpha * self.l) 
            self.x_and_y_count[:, cnt] = X_in_class.sum(0)
               
    def _train_continuous(self, features, targets):  
        # Non binary data
        self.mean = np.zeros((self.M, self.num_classes))
        self.std = np.zeros((self.M, self.num_classes))
        targets_transpose = targets.transpose()[0]
        for cnt, y in enumerate(self.classes):
            s = StandardScaler(with_mean=False)
            x_in_class = features[targets_transpose == y]
            mean = x_in_class.mean(0)
            s.fit(x_in_class)
            std = s.std_
            self.mean[:, cnt] = mean
            self.std[:, cnt] = std

    def _train_multinomial(self, features, targets):
        # Train when all of the features are word counts
        N,M = features.shape
        self.x_given_y = np.zeros((M, self.num_classes))
        self.totals = np.zeros((self.num_classes,1))
        targets_transpose = targets.transpose()[0]
        for cnt, y in enumerate(self.classes):
            self.totals[cnt] = features[(targets==y).flatten(), :].sum()
            x_in_class = features[targets_transpose == y]
            #uniform prior smoothing
            self.x_given_y[:, y] = (x_in_class.sum(0) + 1)/(self.totals[y] + features.shape[1]) 

    def predict(self, observed):
        N,M = observed.shape
        totals = np.zeros((N,self.num_classes))
        # add prior
        totals += np.log(self.py).transpose()
        if not self.continuous:
            logs = np.log(self.x_given_y)
            negs = np.log(1-self.x_given_y)
        if self.bernoulli: 
            totals += observed.dot(logs-negs) + negs.sum(axis=0)
        elif self.multinomial:
            totals += observed.dot(logs)
        else:
            totals += np.log(norm.pdf(observed, loc=self.mean, scale=self.std)).flatten()
        return self.classes_array[np.argmax(totals,1)]

    def score(self, X, y):
        yp = self.predict(X)
        return float((yp == y).sum())/y.shape[0]
