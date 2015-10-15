__author__ = 'Ethan'

import numpy as np

class LeavePOut(object):

	def __init__(self, n, p, shuffle=False):
		'''
		n (int) = number of elements
		p (int) = number of to leave out
		shuffle (bool) = whether or not to shuffle
		'''
		if n <= p: raise Exception("Number of elements must be smaller than p")
		self.n = n
		self.p = p
		self.indices = np.arange(n)
		if shuffle: np.random.shuffle(self.indices)
		self.start_index = 0
		self.end_index = p

	def __iter__(self):
		return self

	def next(self):
		if self.end_index > self.n: raise StopIteration
		train = np.concatenate([self.indices[:self.start_index], self.indices[self.end_index:]])
		test = self.indices[self.start_index:self.end_index]
		self.start_index += 1
		self.end_index += 1
		return train, test