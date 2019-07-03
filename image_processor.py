# import from std lib
import time
from itertools import combinations

# import from third party lib
import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided

# import lib code
from console import Console
from magnetic_memory_strip import MagneticMemoryStrip

# the image processor class
class ImageProcessor:
	def __init__(self, size=24):
		'''
		size:image size square matrix
		'''
		# initialize console
		self.console = Console()
		self.log = self.console.log

		self.setLogState(True)

		# initialize magnetic_memory_strip
		self.MagneticMemoryStrip = []

		# the history
		self.context = []
		self.context_length = 10

		n = size
		for _ in range(n+1):
			self.MagneticMemoryStrip.append(MagneticMemoryStrip())

		# the size
		if type(size) == int:
			self.SIZE = (size, size)

		# constants
		self.MEMORY_PATH = 'memory'
		self.IMAGE_MEMORY_PATH = '{}/images'.format(self.MEMORY_PATH)

		self.dummy_counter = 0

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

		n = self.SIZE[0]
		if self.SIZE[-1] > n:
			n = self.SIZE[-1]

		inference = []

		# kernel size
		for kernel_size in range(3, n+1):
			# the memorystyrip object
			mms = self.MagneticMemoryStrip[kernel_size]

			inference.append({})

			if kernel_size > 3:
				continue

			# get the image features from kernels
			features = self.getKernels(grey, kernel_size)
			a, b, c, d = features.shape

			# get and register all kernels
			for i in range(a):
				for j in range(b):
					# extract feature
					feature = features[i, j]

					if self.dummy_counter == 15:
						# compare to see related properties indices
						start, end = mms.getCloseIndex(feature)

						for x in mms.getData(start, end):
							for xx in x:
								xd = xx[0]
								if xd in inference[-1]:
									inference[-1][xd] += 1

								else:
									inference[-1][xd] = 1

					# save property
					mms.add([image_name, i, j], feature)
		
		if self.dummy_counter == 15:
			for ki, kk in enumerate(inference):
				print(ki + 3)

				for x in kk:
					print('  {}: {}'.format(x, kk[x]))

		# register in memory
		self.addToContext(image_name)
		return

	def addToContext(self, image_path):
		self.context.append(image_path)
		if len(self.context) == self.context_length:
			self.context.pop(0)
		return

	def saveImage(self, image):
		image_name = str(time.time())
		image_path = '{}/{}.jpg'.format(self.IMAGE_MEMORY_PATH, image_name)
		self.console.log('  registering {}'.format(image_path))
		cv2.imwrite(image_path, image)
		return image_name

	def getKernels(self, img, kernel_size):
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