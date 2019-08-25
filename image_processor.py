# import from std lib
import time, os
from itertools import combinations
import threading

# import from third party lib
import cv2
import numpy as np
from matplotlib import pyplot as plt

# import lib code
from console import Console
from memory_line import MemoryLine
from utils import getKernels, resultant, validateFolderPath

# the image processor class
class ImageProcessor:
	def __init__(self, refid, kernel_size=3):
		'''
		refid: reference id
		kernel size: kernel size 
		'''

		# initialize console
		self.console = Console()
		self.log = self.console.log

		self.console.setLogState(True)

		# the image ref id
		self.refid = refid

		# initialize magnetic_memory_strip
		self.image_memory_line = MemoryLine(kernel_size)
		self.kernel_memory_line = MemoryLine(kernel_size)
		

		# the history
		self.context = []
		self.context_length = 10

		self.kernel_size = kernel_size

		# constants
		self.MEMORY_PATH = 'memory'
		self.IMAGE_MEMORY_PATH = '{}/images/{}'.format(self.MEMORY_PATH, refid)

		for folder_path in [self.MEMORY_PATH, self.IMAGE_MEMORY_PATH]:
			validateFolderPath(folder_path)

	def getFeatures(self):
		self.datons = None

		if self.kernel_memory_line.data is None:
			return

		self.datons = np.array([], dtype=np.float64)

		# the sets that all the kernels cluster into
		self.kernel_memory_line.sortAndCluster()

		# save the max and min of a daton possibility
		self.min_daton_indices_memory_line = MemoryLine(1)
		self.max_daton_indices = np.array([], dtype=np.float64) #initialize with zero cos of the memoryline of the start indices

		# function to get the daton( for the sake of threads)
		def collectDaton(start, end):
			index = (np.array(range(start, end), dtype=np.int64),)
			
			data = self.kernel_memory_line.data[index]
			mean, standard_deviation = data.mean(axis=0), data.std(axis=0)
			mean_of_sd = standard_deviation.mean()

			# daton_indices = np.where(standard_deviation <= mean_of_sd)
			daton_indices = np.where(standard_deviation == standard_deviation)

			# the representation of the data in number line space
			daton = mean[daton_indices]
			
			# the resultant of the daton
			daton_id = daton.mean()

			min_daton = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float64) 
			max_daton = 255 + min_daton

			dx = min_daton - 1
			dx[daton_indices] = daton

			# the position of the daton in daton memory
			if len(self.datons) == 0:
				self.datons = np.array([dx], dtype=np.float64)

			else:
				self.datons = np.concatenate((self.datons, [dx]))

			min_daton[daton_indices] = (mean - standard_deviation)[daton_indices]
			min_daton[min_daton < 0] = 0

			max_daton[daton_indices] = (mean + standard_deviation)[daton_indices]
			max_daton[max_daton > 255] = 255
			
			min_euclidean_distance = resultant(min_daton)
			max_euclidean_distance = resultant(max_daton)


			ls = self.console.log_state
			self.console.setLogState(False)
			self.console.log('*\nmean =\n===========\n{}'.format(mean))
			self.console.log('*\nmean of std = {}, std =\n===========\n{}'.format(mean_of_sd, standard_deviation))
			self.console.log('*\nmin_eucld = {}, min_dat =\n========\n{}'.format(min_euclidean_distance, min_daton))
			self.console.log('*\nmax_eucld = {}, max_dat =\n========\n{}'.format(max_euclidean_distance, max_daton))
			self.console.log('*\nstart = {}, end = {}, daton_id = {}\n'.format(start, end, daton_id))
			self.console.setLogState(ls)

			# print(daton_id)
			# print(self.min_daton_indices_memory_line.indices, min_euclidean_distance)
			# print(self.min_daton_indices_memory_line.data, daton_id)
			# print(dx, len(daton_indices[0]), daton.mean())


			# save the starting and ending index of possible letter representation
			self.min_daton_indices_memory_line.add(daton_id, min_euclidean_distance, True)
			self.max_daton_indices = np.append(self.max_daton_indices, max_euclidean_distance)
			
			return daton_id

		# the threads are to be held here
		threads = list(range(len(self.kernel_memory_line.clusters)))
		for i, clx in enumerate(self.kernel_memory_line.clusters):
			start, end = clx

			# procedural
			daton_id = collectDaton(start, end)
			# print(start, end, daton_id)

			# threading
			# threads[i] = threading.Thread(target=collectDaton, args=(start, end,))
			# threads[i].start()

	def findRelatedFeature(self, kernel, kernel_index):
		if self.datons is None or self.min_daton_indices_memory_line.indices is None:
			return

		# get the closest index to start, these three lines must stick as the sorting must affect all
		min_related_index = self.min_daton_indices_memory_line.getRelatedData(kernel_index)
		self.max_daton_indices = self.max_daton_indices[self.min_daton_indices_memory_line.sort_indices]
		self.datons = self.datons[self.min_daton_indices_memory_line.sort_indices]

		# if the min_related_index is not
		if min_related_index == None:
			# print(1)
			return

		# if the starting point is greater than the value itself
		if min_related_index == 0:
			# print(2, kernel_index, self.min_daton_indices_memory_line.indices[min_related_index])
			return self.min_daton_indices_memory_line.data[min_related_index]
		
		# if most similar(starting point) is larger 			
		min_related_index = min_related_index-1 if self.min_daton_indices_memory_line.indices[min_related_index] > kernel_index else min_related_index

		# use this starting point index to get the ending index slice
		mri = min_related_index+2 if self.max_daton_indices[min_related_index] <= kernel_index else min_related_index+1

		max_related_indices = self.max_daton_indices[:mri]

		if len(max_related_indices) == 0:
			# print(3, kernel_index, mri, self.min_daton_indices_memory_line.indices[:mri])
			return

		# the features related
		rang = np.where(max_related_indices >= kernel_index)
				
		# the related fetures and matching daton indices
		related_features = self.min_daton_indices_memory_line.data[rang]

		if len(related_features) < 1:
			# print(4, kernel_index, kernel)
			return

		if len(related_features) == 1:
			# print(5, kernel_index, self.min_daton_indices_memory_line.indices[:mri], self.max_daton_indices, rang, related_features, self.min_daton_indices_memory_line.data)
			return related_features[0]

		daton_indices = self.datons[rang]
		dix = daton_indices.copy()

		# the number of non negative values in each daton
		n = (self.kernel_size**2)  - (np.count_nonzero(daton_indices == -1, axis=tuple(range(len(daton_indices.shape)))[1:]))

		x = np.where(daton_indices == -1)
		daton_indices[x] = kernel[x[1:]]

		# difference between daton and kernel
		ddi = abs(daton_indices - kernel)

		# checking the deviations
		ddi_dev = np.sum(ddi, axis=tuple(range(len(ddi.shape)))[1:]) / n
			
		# getting the size of the daton
		val_index = np.where(ddi_dev == ddi_dev.min())

		ls = self.console.log_state
		self.console.setLogState(False)
		self.console.log('kernel_index = {}, range = {} to {}, daton_id = {}, deviation = {}'.format(
				kernel_index, self.min_daton_indices_memory_line.indices[rang], max_related_indices[rang],
				related_features, ddi
			)
		)
		
		self.console.setLogState(ls)
		# comment to check the kerenel representation value
		# print(kernel_index, related_features[val_index][0])
		# print(dix[val_index][0])
		# print(kernel)
		# print()

		return related_features[val_index][0]

	def getSimilar(self, image, threshold=10, verbose=0):
		# if not verbose:
		# 	return

		if self.image_memory_line.data is None:
			return [], []

		# the deviations
		dev = abs(image - self.image_memory_line.data).mean(axis=tuple(range(len(self.image_memory_line.data.shape)))[1:])
		udev = np.unique(dev)

		if type(threshold) == float:
			threshold = int(threshold * len(udev)) - 1

		if threshold > len(udev)-1:
			threshold = len(udev) - 1

		# the limit of recall
		limit = udev[threshold]

		# the similar images
		similar = self.image_memory_line.indices[dev <= limit][dev[dev <= limit].argsort()]
		similar_ratio = (255 - dev[dev <= limit][dev[dev <= limit].argsort()])/255

		return similar, similar_ratio

	def register(self, image, image_name, verbose=0):
		# save image in memory
		# image_ref = self.saveImage(image, image_name)
		
		# self.image_memory_line.add(image_name, resultant(image))
		image_index = self.image_memory_line.add(image, image_name)
		
		# get the image features from kernels
		kernels = self.getKernels(image)
		a, b, c, d = kernels.shape

		# the alphabet
		if verbose: self.getFeatures()

		# the feture maps to present
		feature_map = np.zeros((a, b))

		# get and register all kernels
		for i in range(a):
			for j in range(b):
				# extract feature
				kernel = kernels[i, j]

				# save property
				kernel_index = self.kernel_memory_line.add(kernel)

				# find similar data
				if not verbose:
					continue

				# print(kernel)

				feature = self.findRelatedFeature(kernel, kernel_index)
				# print(kernel_index, feature, kernel)
				# print()
								
				if feature is None:
					continue
				
				feature_map[i, j] = feature

		# register in memory
		self.addToContext(image_name)
		return feature_map

	def addToContext(self, image_path):
		self.context.append(image_path)
		if len(self.context) == self.context_length:
			self.context.pop(0)
		return

	def saveImage(self, image, image_name=None):
		'''
		saveImage: saves images in the memory directory
			image: matrix of rgb values
		'''
		# the image reference
		image_name = str(time.time()) if image_name is None else image_name
		image_path = '{}/{}.jpg'.format(self.IMAGE_MEMORY_PATH, image_name)
		
		self.console.log('REGISTERING {}'.format(image_path))

		cv2.imwrite(image_path, image)
		return image_name

	def getKernels(self, img, kernel_size=None):
		if kernel_size is None:
			kernel_size = (self.kernel_size, self.kernel_size)

		if type(kernel_size) == int:
			kernel_size = (kernel_size, kernel_size)

		return getKernels(img, kernel_size)