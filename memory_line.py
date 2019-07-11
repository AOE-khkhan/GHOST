# from std lib
from console import Console

# import from third party libraries
import cv2
import numpy as np

# import lib code
from utils import getKernels, load_image, resultant

class SortedList:
	def __init__(self, *args):
		self.elements = list(args)
		self.length = 0

	def append(self, element):
		self.elements.append(element)
		self.elements = sorted(self.elements)
		self.length += 1

	def __repr__(self):
		return str(self.elements)

	def __iter__(self):
		for x in self.elements:
			yield x

	def __len__(self):
		return self.length

class MemoryLine:
	def __init__(self, kernel_size=3, active=False):
		# initialize console
		self.console = Console()
		self.log = self.console.log

		self.console.setLogState(True)

		# for fetching kernels
		self.kernel_size = (kernel_size, kernel_size)

		# if the data will be preprocessed within
		self.active = active

		# get list that sorts itself everytime
		self.indices = SortedList()

		# get the data that holds data
		self.data = None

		# the mean common diff
		self.meancd = None

		# standard dev of common difference
		self.stdcd = None

		# the clusters of indices
		self.clusters = []

	def computeIndex(self, data):
		'''
		data: a 2 dimensional numpy matrix
		'''
		if type(data) == list:
			data = np.array(data)
		return resultant(data)

	def updateCommonDifference(self):
		'''
		get the mean cmmon difference if the indices list was a sequence
		'''
		self.diff = np.diff(np.array(self.indices.elements))
		
		if len(self.diff) < 1:
			return

		# teh stats
		self.meancd = self.diff.mean()
		self.stdcd = self.diff.std()

		# the sets that all the kernels cluster into
		self.clusters = self.getClusters()

	# define the function for closest index using binary search
	def binarySearch(self, needle, sorted_list=None, head=0):
		'''
		binarySearch(sorted_list:list, needle:int, head:int)
			takes in a sorted list and returns the index of an element being serched for
		
		sorted_list: a sorted list of elemets to be searched
		needle: the element that is being looked for
		head: head start index for tracking the right index
		'''

		if sorted_list == None:
			sorted_list = self.indices.elements

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
			return self.binarySearch(needle, sorted_list[:index_of_center_element], head)

		# when middle element is less than the needle
		elif mid_element < needle:
			return self.binarySearch(needle, sorted_list[index_of_center_element+1:], index_of_center_element+head+1)

		#in unforseen circumstances return no index
		else:
			return

	def getRelated(self, needle):
		indx = self.binarySearch(needle)
		if indx != None:
			return self.indices.elements[indx]

	def getClass(self, data_index):
		for cls in self.clusters:
			a, b = cls

			if a <= data_index and data_index < b:
				return cls

	def getClusters(self):
		clusters = []
		
		a = 0
		for i in range(self.indices.length - 1):
			if self.diff[i] > self.meancd or i == self.indices.length - 2:
				b = i+1

				clusters.append((a, b))
				a = 0
		# self.console.log('{} class(es) detected in memeory of length {}'.format(len(clusters), self.indices.length))
		return clusters

	def getData(self, index):
		for image_ref, i, j in self.data[self.indices.elements[index]]:
			yield getKernels(load_image(image_ref), self.kernel_size)[i, j]			
	
	def add(self, data, data_index=None):
		'''
		data: [m x 3] 2 dimensional numpy array
		Goal: add new data to data
		Description: and index is created to label the data saved
		'''

		# initialize data
		if type(self.data) == type(None):
			# self.data = pd.DataFrame({477436715895.2334:np.array([[1, 2, 3]])})
			self.data = {}

		# if data index not available
		if type(data_index) == type(None):
			data_index = self.computeIndex(data)

		elif type(data_index) != int and type(data_index) != float:
			data_index = self.computeIndex(data_index)
			
		# if data already exists
		if data_index not in self.indices.elements:
			self.indices.append(data_index)
			self.data[data_index] = [data]

		else:
			self.data[data_index].append(data)
		
		if self.active:
			# update the mean value
			self.updateCommonDifference()

		return data_index