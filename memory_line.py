# from std lib
from console import Console

# import from third party libraries
import cv2
import numpy as np

# import lib code
from utils import getKernels, load_image, resultant, is_row_in_array, index_row_in_array

class MemoryLine:
	def __init__(self, kernel_size=3):
		# initialize console
		self.console = Console()
		self.log = self.console.log

		self.console.setLogState(True)
		
		# for fetching kernels
		self.kernel_size = (kernel_size, kernel_size)

		# uneven weight on the diff in values
		self.position_weight = np.arange(1, (kernel_size**2) + 1, dtype=np.float64).reshape(self.kernel_size)

		# the memory is none form start
		self.data = self.indices = None

		# the mean common diff
		self.meancd = None

		# standard dev of common difference
		self.stdcd = None

		# the clusters of indices
		self.clusters = np.array([], dtype=np.int64)

	def initializeMemory(self, data, data_index):
		# get the data that holds data
		dtp = np.float64 if type(data) != str else str
		self.data = np.array([data], dtype=dtp)

		# initializer the pegs
		dtp = np.float64 if type(data_index) != str else str
		self.indices = np.array([data_index], dtype=dtp)
		return

	def computeIndex(self, data):
		'''
		data: a 2 dimensional numpy matrix
		'''
		return resultant(data)

	# define the function for closest index using binary search
	def binarySearch(self, needle, sorted_list, head=0):
		'''
		binarySearch(sorted_list:list, needle:int, head:int)
			takes in a sorted list and returns the index of an element being serched for
		
		sorted_list: a sorted list of elemets to be searched
		needle: the element that is being looked for
		head: head start index for tracking the right index
		'''

		# get the length of the sorted list
		length_of_list = len(sorted_list)

		# check if the list is empty
		if length_of_list == 0:
			return	# return no index

		if length_of_list == 1:
			return head

		# get the middle index of the list
		index_of_center_element = length_of_list // 2

		# get the center element
		mid_element = sorted_list[index_of_center_element]

		# if the needle is the middle element
		if mid_element == needle:
			return head + index_of_center_element

		# when middle element is greater than the needle
		elif mid_element > needle:
			li = sorted_list[:index_of_center_element]
			if len(li) == 0:
				return head + index_of_center_element
			else:
				return self.binarySearch(needle, li, head)

		# when middle element is less than the needle
		elif mid_element < needle:
			li = sorted_list[index_of_center_element+1:]
			if len(li) == 0:
				return head + index_of_center_element
			else:
				return self.binarySearch(needle, li, index_of_center_element+head+1)

		#in unforseen circumstances return no index
		else:
			return

	def getRelatedData(self, needle, by_indices=True):
		# sort First!
		self.sortIndices(by_indices)
		
		# get the closet index
		indx = self.binarySearch(needle, self.indices)

		# make sure the highest index of the most related is picked
		if indx == None:
			return

		# all positions that correspond to closest value:
		pos = np.where(self.indices == self.indices[indx])[0]

		# if pos is empty
		if len(pos) == 0:
			return

		return pos[-1]

	def getRelatedIndex(self, needle):
		indx = self.binarySearch(needle)
		if indx != None:
			return self.indices[indx]

	def getClass(self, data_index):
		for cls in self.clusters:
			a, b = cls

			if a <= data_index and data_index < b:
				return cls

	def sortIndices(self, by_indices=True):
		# sort the ids
		if self.data is not None and not by_indices:
			self.sort_indices = self.data.mean(tuple(range(len(self.data.shape)))[1:]).argsort()

		else:
			if self.indices is not None:
				self.sort_indices = self.indices.argsort()
			
			else:
				return

		# get the sorted ids
		self.indices = self.indices[self.sort_indices]
		self.data = self.data[self.sort_indices]
		return

	def sortAndCluster(self):
		# sort the ids
		self.sortIndices(by_indices=False)
		
		# get the common differences and weight the position to know truly cloe ones
		self.diff = (abs(np.diff(np.array(self.data, dtype=np.float64), axis=0)) * self.position_weight).mean(axis=tuple(range(len(self.data.shape)))[1:])

		if len(self.diff) < 1:
			return

		# adding a full std is known to not converge well
		# self.meancd = self.diff.mean()
		self.meancd = self.diff[self.diff > 0].mean()

		# the clusters
		self.clusters = []
		
		a = 0
		for i in range(len(self.diff)):
			# print("class {}: index = {}, diff = {}, m =[{}, {}]\n{}".format(len(self.clusters), self.indices[i+1], self.diff[i], self.diff[self.diff>0].mean(), self.meancd, self.data[i+1]))
			if self.diff[i] >= self.meancd or i == len(self.indices) - 2:
				b = i+1

				# save a cluster
				self.clusters.append((a, b))
				# self.console.log('  class {}: {} element(s)'.format(len(self.clusters), b-a))

				a = b
			# print()
		self.console.log('{} class(es) detected in memeory of length {}'.format(len(self.clusters), len(self.indices)))
		# print(self.meancd, 'mean')
		return

	def getData(self, index):
		for image_ref, i, j in self.data[self.indices[index]]:
			i, j = map(int, [i, j])
			yield getKernels(load_image(image_ref), self.kernel_size)[i, j]			
	
	def add(self, data, data_index=None, force_add=False):
		'''
		data: [m x 3] 2 dimensional numpy array
		Goal: add new data to data
		Description: and index is created to label the data saved
		'''

		# if data index not available
		if data_index is None:
			data_index = resultant(data)

		elif type(data_index) == np.ndarray:
			data_index = resultant(data_index)

		
		if self.data is None:
			self.initializeMemory(data, data_index)
			return data_index

		# if data already exists
		if not force_add:
			indices = index_row_in_array(data, self.data)
			if len(indices) == 1:
				return self.indices[indices[0]]

		self.indices = np.append(self.indices, [data_index])
		self.data = np.concatenate((self.data, [data]))
		return data_index