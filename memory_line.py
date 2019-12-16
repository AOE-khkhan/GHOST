# from std lib
import time
from console import Console

# import from third party libraries
import cv2
import numpy as np
import pandas as pd

# import lib code
from utils import (
	index_row_in_array, TIMESTAMP
)


class MemoryLine:
	def __init__(self, has_experience=False):
		# initialize console
		self.console = Console()
		self.log = self.console.log

		self.console.setLogState(True)
		
		# the memory is none form start
		self.data = self.metadata = None

	def initializeMemory(self, data, timestamp=None):
		# if timestamp not supplied
		if timestamp is None:
			timestamp = time.time()

		# get the data that holds data
		self.data = np.array([data])
		
		# initialize metadata for the actual data
		self.metadata = pd.DataFrame({0: {TIMESTAMP: timestamp}})
		return 0
	
	def add(self, data, timestamp=None, allow_duplicate=False):
		'''
		data: [m x 3] 2 dimensional numpy array
		allow_duplicate: boolean, define strictness for duplicates
		Goal: add new data to data
		Description: and index is created to label the data saved
		'''

		# if time register not specified
		if timestamp is None:
			timestamp = time.time()

		# when the data is first in memory
		if self.data is None:
			return self.initializeMemory(data, timestamp)

		# if data already exists
		if not allow_duplicate:
			indices = index_row_in_array(data, self.data)
			if len(indices):
				return indices[0]

		# the current data index
		data_index = len(self.data)

		# save data and metadata
		self.metadata[data_index] = [timestamp]
		self.data = np.concatenate((self.data, [data]))

		return data_index
