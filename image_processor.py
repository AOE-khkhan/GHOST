# import from std lib
import time, os
from itertools import combinations

# import from third party lib
import cv2
import numpy as np

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
		self.MemoryLine = MemoryLine(kernel_size, active=True)
		

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
		self.features = []
		self.MemoryLineForMinDatonIndex = MemoryLine(kernel_size)
		self.MemoryLineForMaxDatonIndex = MemoryLine(kernel_size)

		for clx in self.MemoryLine.clusters:
			start, end = clx

			dx = []
			for i in range(start, end):
				for x in self.MemoryLine.getData(i):
					dx.append(x)

			dx = np.array(dx)
			m, s = dx.mean(axis=0), dx.std(axis=0)
			ms = s.mean()

			di = len(self.features)
			daton_index = resultant(m)
			self.features.append(daton_index)

			min_daton = np.zeros((self.kernel_size, self.kernel_size))
			min_daton[s <= ms] = m[s <= ms]
			
			min_daton -= s
			min_daton[min_daton < 0] = 0
			
			max_daton = 255 + np.zeros((self.kernel_size, self.kernel_size))
			max_daton[s <= ms] = m[s <= ms]
			
			max_daton += s
			max_daton[max_daton > 255] = 255
			
			# print(min_daton)
			# print(max_daton)

			min_euclidean_distance = resultant(min_daton)
			max_euclidean_distance = resultant(max_daton)
			# print(min_euclidean_distance, max_euclidean_distance)

			# save the starting and ending index of possible letter representation
			self.MemoryLineForMinDatonIndex.add(di, min_euclidean_distance)
			self.MemoryLineForMaxDatonIndex.add(di, max_euclidean_distance)
			

	def validateFolderPath(self, folder_path):
		if not os.path.exists(folder_path):
			os.mkdir(folder_path)

	def toGrey(self, img, r=0.299, g=0.587, b=0.114):
		return np.add(b*img[:, :, 0], g*img[:, :, 1], r*img[:, :, 2])

	def register(self, image):
		# get the grey version of image
		grey = self.toGrey(image, 1/3, 1/3, 1/3)

		# save image in memory
		image_ref = self.saveImage(image)

		# get the image features from kernels
		features = self.getKernels(grey)
		a, b, c, d = features.shape

		# the alphabet
		self.getFeatures()

		# get and register all kernels
		for i in range(a):
			for j in range(b):
				# extract feature
				feature = features[i, j]

				# save property
				feature_index = self.MemoryLine.add([image_ref, i, j], feature)
				
				min_related = self.MemoryLineForMinDatonIndex.getRelatedData(feature_index)
				max_related = self.MemoryLineForMaxDatonIndex.getRelatedData(feature_index)

				if min_related == max_related == None:
					continue

				related = set(min_related).intersection(set(max_related))
				print(feature)

				for r in related:
					print(self.features[r])
				print()

		# register in memory
		self.addToContext(image_ref)
		return

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
		
		self.console.log('  registering {}\n'.format(image_path))

		cv2.imwrite(image_path, image)
		return image_path

	def getKernels(self, img, kernel_size=None):
		if type(kernel_size) == type(None):
			kernel_size = (self.kernel_size, self.kernel_size)

		if type(kernel_size) == int:
			kernel_size = (kernel_size, kernel_size)

		return getKernels(img, kernel_size)