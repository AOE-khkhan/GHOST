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
from utils import getKernels, resultant

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

		# initialize magnetic_memory_strip
		self.kernel_memory_line = MemoryLine(kernel_size)
		

		# the history
		self.context = []
		self.context_length = 10

		self.kernel_size = kernel_size

		# constants
		self.MEMORY_PATH = 'memory'
		self.IMAGE_MEMORY_PATH = '{}/images/{}'.format(self.MEMORY_PATH, refid)

		for folder_path in [self.MEMORY_PATH, self.IMAGE_MEMORY_PATH]:
			self.validateFolderPath(folder_path)

	def getFeatures(self):

		self.features = np.array([])

		# the sets that all the kernels cluster into
		self.kernel_memory_line.sortAndCluster()

		# save the max and min of a daton possibility
		self.min_daton_indices_memory_line = MemoryLine(1)
		self.max_daton_indices = np.array([0.0]) #initialize with zero cos of the memoryline of the start indices

		# function to get the daton( for the sake of threads)
		def collectDaton(start, end):
			index = (np.array(range(start, end), dtype=np.int64),)
			
			data = self.kernel_memory_line.data[index]
			mean, standard_deviation = data.mean(axis=0), data.std(axis=0)
			mean_of_sd = standard_deviation.mean()

			daton_indices = np.where(standard_deviation <= mean_of_sd)

			# the representation of the data in number line space
			daton = mean[daton_indices]
			
			# the resultant of the daton
			daton_id = resultant(daton)

			# the position of the daton in daton memory
			if len(self.features) == 0:
				self.features = np.array([daton_id])

			else:
				self.features = np.concatenate((self.features, [daton_id]))

			min_daton = np.zeros((self.kernel_size, self.kernel_size)) 
			max_daton = 255 + min_daton

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


			# save the starting and ending index of possible letter representation
			self.min_daton_indices_memory_line.add(daton_id, min_euclidean_distance)
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
	
	def findRelated(self, feature_index):
		# get the closest index to start
		min_related_index = self.min_daton_indices_memory_line.getRelatedData(feature_index)
		self.max_daton_indices = self.max_daton_indices[self.min_daton_indices_memory_line.sort_indices]

		if min_related_index == None:
			return

		# use this starting point index to get the ending index slice
		if self.max_daton_indices[min_related_index] <= feature_index:
			max_related_indices = self.max_daton_indices[:min_related_index+1]

		else:
			max_related_indices = self.max_daton_indices[:min_related_index]

		if len(max_related_indices) == 0:
			return

		# the features related
		rang = np.where(max_related_indices >= feature_index)
		related = self.min_daton_indices_memory_line.indices[rang]

		# print(feature_index, related, max_related_indices[rang])
		return self.min_daton_indices_memory_line.data[rang]

	def getSimilar(self, image):

		return

	def validateFolderPath(self, folder_path):
		if not os.path.exists(folder_path):
			os.mkdir(folder_path)

	def toGrey(self, img, r=0.299, g=0.587, b=0.114):
		s = img.shape
		if len(s) == 2 or (len(s) == 3 and s[-1] < 3):
			return img
		return np.add(b*img[:, :, 0], g*img[:, :, 1], r*img[:, :, 2])

	def register(self, image, verbose=0):
		# get the grey version of image
		grey = self.toGrey(image, 1/3, 1/3, 1/3)

		# save image in memory
		image_ref = self.saveImage(image)

		# get the image features from kernels
		features = self.getKernels(grey)
		a, b, c, d = features.shape

		# the alphabet
		if verbose: self.getFeatures()

		# the feture maps to present
		feature_maps = np.expand_dims(np.zeros((a, b)), axis=0)

		# get and register all kernels
		for i in range(a):
			for j in range(b):
				# extract feature
				feature = features[i, j]

				# print(feature)

				# save property
				feature_index = self.kernel_memory_line.add(feature)
				
				# find similar data
				if not verbose:
					continue

				related_classes = self.findRelated(feature_index)
				
				if type(related_classes) == type(None) or len(related_classes) == 0:
					continue

				# the last state of the feature maps
				fm = feature_maps.copy()
				feature_maps = []

				print(len(related_classes))
				
				# effect the change accordingly for every posibility
				for rep in related_classes:
					fmx = fm.copy()
					print(rep, i, j, fmx.shape)

					fmx[:, i, j] = rep
					feature_maps.extend(fmx)

				feature_maps = np.array(feature_maps)

		# register in memory
		self.addToContext(image_ref)
		return feature_maps

	def addToContext(self, image_path):
		self.context.append(image_path)
		if len(self.context) == self.context_length:
			self.context.pop(0)
		return

	def saveImage(self, image):
		'''
		saveImage: saves images in the memory directory
			image: matrix of rgb values
		'''
		# the image reference
		image_name = str(time.time())
		image_path = '{}/{}.jpg'.format(self.IMAGE_MEMORY_PATH, image_name)
		
		self.console.log('registering {}'.format(image_path))

		cv2.imwrite(image_path, image)
		return image_path

	def getKernels(self, img, kernel_size=None):
		if type(kernel_size) == type(None):
			kernel_size = (self.kernel_size, self.kernel_size)

		if type(kernel_size) == int:
			kernel_size = (kernel_size, kernel_size)

		return getKernels(img, kernel_size)