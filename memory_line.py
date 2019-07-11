# from std lib
from console import Console

# import from third party libraries
import cv2
import numpy as np

# import lib code
from utils import getKernels, load_image

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

class MemoryLine:
	def __init__(self):
		# initialize console
		self.console = Console()
		self.log = self.console.log

		self.console.setLogState(True)

		# get list that sorts itself everytime
		self.indices = SortedList()

		# get the data that holds data
		self.data = None

		# the mean common diff
		self.meancd = None

		# standard dev of common difference
		self.stdcd = None

		# the clusters of indices
		self.classes = []

	def computeIndex(self, data):
		'''
		data: a 2 dimensional numpy matrix
		'''
		DIM = 3 #length of the info collected from each pixel - x, y, color
		if type(data) == list:
			data = np.array(data)

		d = data.copy()
		s = np.sum(data)
		return round(s, 4)

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
		self.classes = self.getClasses()

		self.getFeatures()

	# define the function for closest index using binary search
	def binarySearch(self, needle, sorted_list=None, head=0):
		'''
		binary_search(sorted_list:list, needle:int, head:int)
			takes in a sorted list and returns the index of an element being serched for
		
		sorted_list: a sorted list of elemets to be searched
		needle: the element that is being looked for
		head: head start index for tracking the right index
		'''

		if sorted_list == None:
			sorted_list = self.data.indices

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
			return binary_search(sorted_list[:index_of_center_element], needle, head)

		# when middle element is less than the needle
		elif mid_element < needle:
			return binary_search(sorted_list[index_of_center_element+1:], needle, index_of_center_element+head+1)

		#in unforseen circumstances return no index
		else:
			return

	def getClass(self, data_index):
		for cls in self.classes:
			a, b = cls

			if a <= data_index and data_index < b:
				return cls

	def getClasses(self):
		classes = []
		
		a = 0
		for i in range(self.indices.length - 1):
			if self.diff[i] > self.meancd or i == self.indices.length - 2:
				b = i+1

				classes.append((a, b))
				a = 0
		# self.console.log('{} class(es) detected in memeory of length {}'.format(len(classes), self.indices.length))
		return classes

	def getData(self, index):
		for image_ref, i, j in self.data[self.indices.elements[index]]:
			yield getKernels(load_image(image_ref), (3, 3))[i, j]

	def getFeatures(self):
		self.getFeatures = []
		for clx in self.classes:
			start, end = clx

			dx = []
			for i in range(start, end):
				for x in self.getData(i):
					dx.append(x)

			dx = np.array(dx)
			m, s = dx.mean(axis=0), dx.std(axis=0)
			ms = s.mean()
			
	
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
			
		# update the mean value
		self.updateCommonDifference()

		return data_index

	def runTests(self):
		'''
		run test on the module
		OrderTest: To see if the sorting list sorts the indices
		SearchTest: To find a most related values
		'''
		def orderTest(data):
			# get the data index
			di = self.computeIndex(data)
			print('saving data of index =', di, end='\n\n')

			# add data to strip
			self.add(data)
			print('MMS =', self.indices, end='\n\n')

		print('testing for MMS order\n' + '='*20)
			
		# initialize data
		data = np.ones((3, 3))
		orderTest(data)

		data = np.ones((1, 3))
		orderTest(data)

		print(self.data)
		
		print('testing for MMS search\n' + '='*20)
		import random

		for i in range(1000):
			data = np.random.randint(256, size=(random.randint(1, 10), 3))
			self.add(data)
		
		a, b = int(self.indices.elements[0]), int(self.indices.elements[-1])
		r_idx = 7728#random.randint(a, b)

		print('min = {}, max = {}, length = {}, random id = {}'.format(a, b, self.indices.length, r_idx))
		
		ids = self.getCloseIndex(r_idx)
		print('random id = {}, index of random id = {}'.format(r_idx, ids))