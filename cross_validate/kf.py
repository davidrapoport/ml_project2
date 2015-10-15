__author__ = 'Ethan'

import numpy as np

class KFold(object):

	def __init__(self, n, num_folds, shuffle=False):
		'''
		n (int) = number of elements
		folds (int) = number of folds (must be > 1)
		shuffle (bool) = whether or not to shuffle #TODO
		'''
		if num_folds < 2: raise Exception("Number of folds in KFold must be > 1")
		if n < num_folds: raise Exception("Size must be greater than number of folds")
		self.n = n
		self.num_folds = num_folds
		self.indices = np.arange(n)
		if shuffle: np.random.shuffle(self.indices)
		self.folds = np.array_split(self.indices, num_folds)
		self.test_index = 0

	def __iter__(self):
		return self

	def next(self):
		if self.test_index >= self.num_folds: raise StopIteration
		all_indices = range(0,self.num_folds)
		train = np.concatenate([self.folds[x] for x in all_indices[:self.test_index] + all_indices[self.test_index+1:]])
		test = self.folds[all_indices[self.test_index]]
		self.test_index += 1
		return train, test