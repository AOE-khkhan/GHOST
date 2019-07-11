# import from third party libraries
import cv2
import numpy as np

# import pandas as pd

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

class MemoryLine:
	def __init__(self):
		# get list that sorts itself everytime
		self.indices = SortedList()

		# get the memory that holds data
		self.memory = None

		# the mean common diff
		self.meancd = None

		# standard dev of common difference
		self.stdcd = None

	def computeIndex(self, data):
		'''
		data: a 2 dimensional numpy matrix
		'''
		DIM = 3 #length of the info collected from each pixel - x, y, color
		if type(data) == list:
			data = np.array(data)

		d = data.copy()
		s = np.sum(data)
		return s

	def getData(self, start, end=None):
		if end == None:
			yield self.memory[self.indices.elements[start]]

		else:
			for x in range(start, end):
				yield self.memory[self.indices.elements[x]]

	def updateCommonDifference(self):
		'''
		get the mean cmmon difference if the indices list was a sequence
		'''
		diff = np.diff(np.array(self.indices.elements))
		
		if len(diff) < 1:
			return

		self.meancd = diff.mean()
		self.stdcd = diff.std()

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
			sorted_list = self.memory.elements

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

	def getCloseIndex(self, idx, padd=-1):
		def validate(a, b):
			if a < 0:
				a = 0

			if b >= length:
				b = length - 1

			return a, b

		# if data not established
		if self.meancd == None or self.meancd == 0.0:
			return 

		# data type is array instead of int
		if type(idx) != int:
			idx = self.computeIndex(idx)

		# the data index
		n = self.binarySearch(idx)
		
		# the search range		
		limit = self.meancd

		# initial values
		a, li, state = None, [True, True], [n, n+1]

		while True:
			if state[0] == state[1] == False:
				break

			for i in range(2):
				if state[i]:
					if i == 0:
						if li[i] > 0 and abs(self.indices.elements[li[i] - 1] - self.indices.elements[li[i]]) <= limit:
							li[i] -= 1
						
						else:
							state[i] = False

					if i == 1:
						if li[i] < self.indices.length-1 and abs(self.indices.elements[li[i] + 1] - self.indices.elements[li[i]]) <= limit:
							li[i] += 1

						else:
							state[i] = False

			a, b = li

		if a == None:
			return

		else:
			return a, b


	def add(self, data, data_index=None):
		'''
		data: [m x 3] 2 dimensional numpy array
		Goal: add new data to memory
		Description: and index is created to label the data saved
		'''

		# initialize memory
		if type(self.memory) == type(None):
			# self.memory = pd.DataFrame({477436715895.2334:np.array([[1, 2, 3]])})
			self.memory = {}

		# if data index not available
		if type(data_index) == type(None):
			data_index = self.computeIndex(data)

		elif type(data_index) != int and type(data_index) != float:
			data_index = self.computeIndex(data_index)
			
		# if data already exists
		if data_index not in self.indices.elements:
			self.indices.elements.append(data_index)
			self.memory[data_index] = [data]

		else:
			self.memory[data_index].append(data)
			
		# update the mean value
		self.updateCommonDifference()


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

		print(self.memory)
		
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