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
from utils import getKernels, resultant, validateFolderPath, is_row_in_array

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

	def getSimilar(self, image, threshold=10, verbose=0):
		# if not verbose:
		# 	return

		if self.image_memory_line.data is None:
			return [], []

		# the deviations
		dev = np.array([self.compare(image, img) for img in self.image_memory_line.data], dtype=np.float64)
		udev = np.unique(dev)[::-1]

		if type(threshold) == float:
			base = threshold

		else:
			if threshold > len(udev)-1:
				threshold = len(udev) - 1

			# the base of similarity
			base = udev[threshold]

		# the similar images
		similar = self.image_memory_line.indices[dev.argsort()[::-1]]
		similar_ratio = dev[dev.argsort()[::-1]]
		return similar[similar_ratio >= base], similar_ratio[similar_ratio >= base]

	def run(self, image, image_name, verbose=0):
		cls = []
		image_unique = np.unique(image)

		# # self.image_memory_line.add(image_name, resultant(image))
		# _ = self.image_memory_line.add(image, image_name)

		# get the objects
		# yield self.getSimilar(image)

		for i in image_unique:
			m = abs(i - image_unique).flatten()
			m = m.mean() - m.std()/2

			a = 0 if i - m < 0 else i - m
			b = 255 if i + m > 255 else i + m

			img = ((image >= a) & (image <= b)).astype(np.int8)

			valid = False
			if len(cls) == 0:
				cls = np.array([img], dtype=np.int8)
				last_a, last_b = a, b
				valid = True

			else:
				if a > (last_a + last_b)/2 and (not is_row_in_array(img, cls)):
					cls = np.concatenate((cls, [img]))
					last_a, last_b = a, b
					valid = True

			if not valid:
				continue

			imgx = img.copy()
			ret, labels = cv2.connectedComponents(img)
			# labeled_img = imshow_components(labels)

			# register in memory
			self.addToContext(image_name)

			for label in range(1, ret):
				pos = np.where(labels == label)
				ar1, ar2 = pos

				x1, x2 = min(ar1), max(ar1)+1
				y1, y2 = min(ar2), max(ar2)+1

				print(ar1.mean(), ar2.mean())

				img_objx = imgx.copy()
				img_objx[np.where(labels != label)] = 0
				img_objx[np.where(labels == label)] = 1

				img_objx = img_objx[x1:x2, y1:y2]

				x, y = img_objx.shape

				img_obj = np.zeros((30, 30), dtype=np.int8)
				img_obj[0:x, 0:y] = img_objx

				# self.image_memory_line.add(image_name, resultant(image))
				_ = self.image_memory_line.add(img_obj, image_name)
				
				# get the objects
				yield self.getSimilar(img_obj)

	def compare(self, img_1, img_2):
		def getSubImage(img):
			pos = np.where(img == 1)
			ar1, ar2 = pos

			x1, x2 = min(ar1), max(ar1)+1
			y1, y2 = min(ar2), max(ar2)+1

			return img[x1:x2, y1:y2]

		img_1 = getSubImage(img_1)
		img_2 = getSubImage(img_2)
		
		c11, c12 = img_1.shape
		c21, c22 = img_2.shape

		r = c21 if c21 < c11 else c11
		c = c22 if c22 < c12 else c12

		img_1 = cv2.resize(img_1.astype(np.uint8), (c, r), interpolation=cv2.INTER_AREA).astype(np.int8)
		img_2 = cv2.resize(img_2.astype(np.uint8), (c, r), interpolation=cv2.INTER_AREA).astype(np.int8)

		img_1a, img_1b = np.where(img_1 == 1)
		img_1a, img_1b = int(img_1a.mean()), int(img_1b.mean())

		img_2a, img_2b = np.where(img_2 == 1)
		img_2a, img_2b = int(img_2a.mean()), int(img_2b.mean())

		d, f = img_1.shape
		g, h = img_2.shape

		if img_1a <= img_2a:
			img_1x1 = img_2a - img_1a
			img_1x2 = img_1x1 + d

			img_2x1, img_2x2 = 0, g

		else:
			img_2x1 = img_1a - img_2a
			img_2x2 = img_2x1 + g

			img_1x1, img_1x2 = 0, d


		if img_1b <= img_2b:
			img_1y1 = img_2b - img_1b
			img_1y2 = img_1y1 + f

			img_2y1, img_2y2 = 0, h

		else:
			img_2y1 = img_1b - img_2b
			img_2y2 = img_2y1 + h

			img_1y1, img_1y2 = 0, f

		r = max((img_1x2, img_2x2))
		c = max((img_1y2, img_2y2))

		# print(
		# 	img_1x1, img_1x2, img_1y1, img_1y2, '=>', img_2x1, img_2x2, img_2y1, img_2y2, '=>', d,g, f,h, [r, c],
		# 	[img_1a, img_1b], [img_2a, img_2b]
		# )

		# the new images
		img_n1, img_n2 = np.zeros((r, c), dtype=np.int8), np.zeros((r, c), np.int8)

		# the new images
		img_n1[img_1x1:img_1x2, img_1y1:img_1y2] = img_1
		img_n2[img_2x1:img_2x2, img_2y1:img_2y2] = img_2

		img_b = img_n1.copy()
		img_b[img_n2 == 0] = 0

		m = 255 if np.amax(img_n1) > 1 or np.amax(img_n2) > 1 else 1

		z1 = (m - abs(img_b - img_n2).mean()) / m
		z2 = (m - abs(img_b - img_n1).mean()) / m
		z3 = (m - abs(img_n2 - img_n1).mean()) / m
		
		return 0.5*(z1 + z2)

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

