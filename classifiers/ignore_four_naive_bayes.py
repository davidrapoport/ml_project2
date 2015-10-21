from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
import pdb, math
import numpy as np

class IgnoreOtherNaiveBayes(BaseEstimator):

    def __init__(self, alpha=1.0, l=2.0, multinomial=True, cnb=False, wcnb=False):
        '''
        alpha: alpha weighting factor
        l: used in BernoulliNB. weight = (w+alpha)/(total +alpha*l)
        cnb: Flag for whether or not to use complimentary multinomial NB
        wcnb: Flag for whether to perform weight normalization and compl NB
        '''
        self.l = l 
        self.alpha = alpha
        self.multinomial = multinomial
        self.cnb = cnb
        self.wcnb = wcnb
        
    def get_params(self, *args, **kwargs):
        return {"l":self.l, "alpha":self.alpha, "multinomial":self.multinomial, "cnb":self.cnb, "wcnb": self.wcnb}

    def fit(self, features, targets):
        self.classes = list(np.unique(targets))
        self.num_classes = len(self.classes) 
        N,M = features.shape
        self.M = M
        
        self.classes_array = np.zeros((self.num_classes,1))
        self.classes_array[:,0]=self.classes

        self.ignore_class = 3
        self.num_classes -= 1
        self.classes.remove(3)


        # Compute p(yi) for all classes
        self.py = np.zeros((self.num_classes,1)) #prob(yi)
        self.num_y = np.zeros((self.num_classes,1))
        for cnt,y in enumerate(self.classes):
            self.num_y[cnt] = (targets==y).sum()
            self.py[cnt] = self.num_y[cnt]/N
        
        self._train_multinomial(features, targets)


    def _train_multinomial(self, features, targets):
        # Train when all of the features are word counts
        N,M = features.shape
        self.x_given_y = np.zeros((M, self.num_classes))
        self.totals = np.zeros((self.num_classes,1))
        other = features[(targets==3).flatten(), :]
        other_targets = targets[(targets==3).flatten()]
        targets = targets[(targets!=3).flatten()]
        features = features[(targets != 3).flatten(), :]
        targets_transpose = targets
        if self.cnb or self.wcnb:
            for cnt, y in enumerate(self.classes):
                self.totals[cnt] = features[(targets==y).flatten(), :].sum()
            self.totals_not_in_c = np.zeros((self.num_classes,1))
            for cnt, y in enumerate(self.classes):
                for i in range(1,self.num_classes):
                    self.totals_not_in_c[cnt] += self.totals[(y+i)%self.num_classes]
                x_not_in_class = features[targets_transpose != y, :]
                self.x_given_y[:, y] = (np.log(x_not_in_class.sum(0) + self.alpha) - 
                                        np.log(self.totals_not_in_c[cnt] + features.shape[1]*self.alpha))
                if self.wcnb:
                    self.x_given_y[:,y] = self.x_given_y[:,y]/np.absolute(self.x_given_y[:,y]).sum()
        else:
            for cnt, y in enumerate(self.classes):
                self.totals[cnt] = features[(targets==y).flatten(), :].sum()
                x_in_class = features[targets_transpose == y, :]
                #uniform prior smoothing
                self.x_given_y[:, y] = np.log(x_in_class.sum(0) + self.alpha) - np.log(self.totals[cnt] + features.shape[1]*self.alpha) 
        
        N,Mp = other.shape
        totals = np.zeros((N,self.num_classes))
        totals += np.log(self.py).transpose()
        if not self.cnb and not self.wcnb:
            totals += other.dot(self.x_given_y)
        else:
            totals -= other.dot(self.x_given_y)
        # Find the threshold for which if none of the predictions are greater than this then predict 3
        m = totals.max(1)
        self.threshold = totals.sum()/m.size
        
    def predict(self, observed):
        N,M = observed.shape
        totals = np.zeros((N,self.num_classes))
        # add prior
        totals += np.log(self.py).transpose()
        if not self.cnb and not self.wcnb:
            totals += observed.dot(self.x_given_y)
        else:
            totals -= observed.dot(self.x_given_y)
        max_vals = totals.max(1)
        others = max_vals < self.threshold
        predictions = self.classes_array[np.argmax(totals,1)]
        predictions[others] = 3
        return predictions

    def score(self, X, y):
        yp = self.predict(X).ravel()
        return float((yp == y).sum())/y.shape[0]

