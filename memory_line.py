# from std lib
from console import Console

# import from third party libraries
import cv2
import numpy as np

# import lib code
from utils import getKernels, load_image, resultant

class MemoryLine:
	def __init__(self, kernel_size=3):
		# initialize console
		self.console = Console()
		self.log = self.console.log

		self.console.setLogState(True)

		# for fetching kernels
		self.kernel_size = (kernel_size, kernel_size)

		# the memory is none form start
		self.data = self.indices = None

		# the mean common diff
		self.meancd = None

		# standard dev of common difference
		self.stdcd = None

		# the clusters of indices
		self.clusters = np.array([])

	def initializeMemory(self, data, data_index):
		# get the data that holds data
		self.data = np.array([data])

		# initializer the pegs
		self.indices = np.array([data_index])
		return

	def computeIndex(self, data):
		'''
		data: a 2 dimensional numpy matrix
		'''
		if type(data) == list:
			data = np.array(data)
		return resultant(data)

	# define the function for closest index using binary search
	def binarySearch(self, needle, sorted_list=None, head=0):
		'''
		binarySearch(sorted_list:list, needle:int, head:int)
			takes in a sorted list and returns the index of an element being serched for
		
		sorted_list: a sorted list of elemets to be searched
		needle: the element that is being looked for
		head: head start index for tracking the right index
		'''

		if type(sorted_list) == type(None):
			sorted_list = self.indices

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

	def getRelatedData(self, needle):
		# sort First!
		self.sortIndices()
		
		# get the closet index
		return self.binarySearch(needle)

	def getRelatedIndex(self, needle):
		indx = self.binarySearch(needle)
		if indx != None:
			return self.indices[indx]

	def getClass(self, data_index):
		for cls in self.clusters:
			a, b = cls

			if a <= data_index and data_index < b:
				return cls

	def sortIndices(self):
		# sort the ids
		self.sort_indices = self.indices.argsort()

		# get the sorted ids
		self.indices = self.indices[self.sort_indices]
		return

	def sortAndCluster(self):
		# sort the ids
		self.sortIndices()

		# get the common differences
		self.diff = np.diff(np.array(self.indices))
		# self.diff = self.diff[self.diff > 0]
		
		if len(self.diff) < 1:
			return

		# teh stats
		self.meancd = self.diff.mean()

		# the clusters
		self.clusters = []
		
		a = 0
		for i in range(len(self.indices) - 1):
			if self.diff[i] > self.meancd or i == len(self.indices) - 2:
				b = i+1

				self.clusters.append((a, b))
				a = b
		self.console.log('{} class(es) detected in memeory of length {}'.format(len(self.clusters), len(self.indices)))
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
		if type(data_index) == type(None):
			data_index = self.computeIndex(data)

		elif type(data_index) != int and type(data_index) != float:
			data_index = self.computeIndex(data_index)
		
		if type(self.data) == type(None):
			self.initializeMemory(data, data_index)
			return data_index

		# if data already exists
		if not force_add and data in self.data:
			return self.indices[np.where(self.data == data)[0][0]]

		self.indices = np.append(self.indices, [data_index])
		self.data = np.concatenate((self.data, [data]))
		
		return data_index