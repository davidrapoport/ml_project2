from scipy.stats import norm
import pdb
import numpy as np

def NaiveBayes(object)

    def __init__(self, l=1.0, alpha=2.0, multinomial=False, continuous=False, bernoulli=True):
        self.l = l 
        self.alpha = alpha
        self.multinomial = multinomial
        self.continuous = continuous
        self.bernoulli = bernoulli
        
    def train(self, features, targets):
        self.classes = np.unique(targets)
        self.num_classes = len(self.classes) 
        N,M = features.shape
        self.M = M

        # Compute p(yi) for all classes
        self.py = np.zeros((self.num_classes,1)) #prob(yi)
        self.num_y = np.zeros((self.numclasses,1))
        for cnt,y in enumerate(self.num_classes):
            self.num_y[cnt] = np.sum(targets==y)[0]
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
        targets_transpose = targets.transpose()
        for feature_num, feature_col in enumerate(feature_transpose):
            if np.logical_or(feature_col == 0, feature_col == 1).all():
                # Binary data
                for cnt, y in enumerate(self.classes):
                    indices = np.logical_and(feature_col == 1, targets_transpose == y)
                    self.x_given_y[feature_num, cnt] = (np.sum(indices) + self.l)/
                                                    (self.num_y[cnt] + self.alpha * self.l) 
                    self.x_and_y_count[feature_num, cnt] = np.sum(indices)
               
    def _train_continuous(self, features, targets):  
        # Non binary data
        self.mean = np.zeros((self.M, self.num_classes))
        self.std = np.zeros((self.M, self.num_classes))
        for feature_num, feature_col in enumerate(features.transpose()):
            for cnt, y in enumerate(self.classes):
                x_in_class = feature_col[targets_transpose == y]
                mean = x_in_class.mean()
                std = x_in_class.std()
                if not x_in_class:
                    mean = feature_col.mean()
                    std = feature_col.std()
                self.mean[feature_num, cnt] = mean
                self.std[feature_num, cnt] = std

    def _train_multinomial(self, features, targets):
        # Train when all of the features are word counts
        self.x_given_y = np.zeros((M, self.num_classes))
        self.totals = np.zeros((self.num_classes,1))
        for cnt, y in enumerate(self.classes):
            self.totals[i] = np.sum(features[(targets==y).flatten(), :])
        for feature_num, feature_col in enumerate(features.transpose()):
            for cnt, y in enumerate(self.classes):
                x_in_class = feature_col[targets_transpose == y]
                self.x_given_y = (np.sum(x_in_class) + self.l)/
                                (self.totals[y] + self.alpha*self.l)

    def predict(self, features)

