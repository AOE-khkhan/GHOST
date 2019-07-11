# import from std lib
import time, os
from itertools import combinations

# import from third party lib
import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided

# import lib code
from console import Console
from memory_line import MemoryLine

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

		self.setLogState(True)

		# initialize magnetic_memory_strip
		self.MemoryLine = MemoryLine()

		# the history
		self.context = []
		self.context_length = 10

		self.kernel_size = kernel_size

		# constants
		self.MEMORY_PATH = 'memory'
		self.IMAGE_MEMORY_PATH = '{}/images/{}'.format(self.MEMORY_PATH, refid)

		for folder_path in [self.MEMORY_PATH, self.IMAGE_MEMORY_PATH]:
			self.validateFolderPath(folder_path)

		self.dummy_counter = 0

	def validateFolderPath(self, folder_path):
		if not os.path.exists(folder_path):
			os.mkdir(folder_path)

	def setLogState(self, state):
		return self.console.setLogState(state)

	def toggleLogState(self):
		return self.console.toggleLogState()

	def load_image(self, image_path):
		image = cv2.imread(image_path)
		# image = image[...,::-1]
		return image

	def toGrey(self, img, r=0.299, g=0.587, b=0.114):
		return cv2.add(b*img[:, :, 0], g*img[:, :, 1], r*img[:, :, 2])

	def register(self, image):
		self.dummy_counter += 1

		# get the grey version of image
		grey = self.toGrey(image, 1/3, 1/3, 1/3)

		# save image in memory
		image_name = self.saveImage(image)

		# get the image features from kernels
		features = self.getKernels(grey)
		a, b, c, d = features.shape

		# get and register all kernels
		for i in range(a):
			for j in range(b):
				# extract feature
				feature = features[i, j]

				# save property
				self.MemoryLine.add([image_name, i, j], feature)
		
		if self.dummy_counter == 15:
			pass

		# register in memory
		self.addToContext(image_name)
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
		return image_name

	def getKernels(self, img, kernel_size=None):
		if type(kernel_size) == type(None):
			kernel_size = (self.kernel_size, self.kernel_size)

		if type(kernel_size) == int:
			kernel_size = (kernel_size, kernel_size)

		return as_strided(
			img,
			shape=(
				img.shape[0] - kernel_size[0] + 1,  # The feature map is a few pixels smaller than the input
				img.shape[1] - kernel_size[1] + 1,
				kernel_size[0],
				kernel_size[1],
			),
			strides=(
				img.strides[0],
				img.strides[1],
				img.strides[0],  # When we move one step in the 3rd dimension, we should move one step in the original data too
				img.strides[1],
			),
			writeable=False,  # totally use this to avoid writing to memory in weird places
		)