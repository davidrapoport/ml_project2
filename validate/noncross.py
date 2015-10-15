__author__ = 'Ethan'

import numpy as np

class NonCross(object):

	def __init__(self, n, test_size, shuffle=False):
		'''
		n (int) = number of elements
		test_size (int) = float between 0.0 and 1.0 indicating what percent of the total array to allocate to testing
		shuffle (bool) = whether or not to shuffle
		'''
		if test_size >= 1.0 or test_size <= 0.0: raise Exception("Failed to meet requirement: 0.0 < test_size < 1.0")
		self.n = n
		self.test_size = test_size
		self.indices = np.arange(n)
		if shuffle: np.random.shuffle(self.indices)
		self.test_start = n - int(test_size*n)
		self.train = self.indices[:self.test_start]
		self.test = self.indices[self.test_start:]
		self.done = False

	def get_split(self):
		return self.train, self.test