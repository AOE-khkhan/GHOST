# import from std lib
import time
from collections import defaultdict

# import from third party lib
import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.metrics.pairwise import cosine_similarity as similarity_function

# import lib code
from context import Context
from console import Console
from memory_line import MemoryLine

cosine_similarity = lambda X, Y, dense_output=True: similarity_function(X+10e-10, Y+10e-10, dense_output)

# the image processor class
class ImageProcessor:
	def __init__(self, cortex, context_maxlength=10):
		'''
		cortex: object reference of the agent cortex
		'''

		# initialize console
		self.console = Console()
		self.log = self.console.log
		self.console.setLogState(True)

		# the central processor object
		self.cortex = cortex
		self.cortex.image_processor = self

		# the history
		self.context = Context(context_maxlength)

		# initialize magnetic_memory_strip
		self.image_memory_line = MemoryLine()

	def discoverSegments(self, image, kernel_size):
		'''
			return segemets gotten pulse from kernels due to contrast(back and fore gound)

		image - np.array of image
		kernel is a ratio of image size
		'''
		# get kernels drived from size
		kernels = self.getKernels(image, kernel_size)

		# kernels along the horizntal
		for kernelx in kernels:
			#kernels along the vertical
			for kernel in kernelx:

				#the pixels in kernel
				kernel_pixels = kernel.flatten()

				#the starting pixel
				pixel = kernel_pixels.min()

				#distribution of kernel pixels
				distr = kernel_pixels
				
				#the barrier to cross
				threshold = (distr.mean() + (distr.min() / 2))
				
				#outlier detection
				outlier_check = kernel_pixels > threshold

				#if outlier not present, high end class
				if not outlier_check.any():
					continue

				#add lower class to the classes found
				yield pixel, kernel_pixels[~outlier_check].max()

				#get the outliers
				high_end = kernel_pixels[outlier_check]

				#add to the classes found
				yield high_end.min(), high_end.max()

	def getKernels(self, img, kernel_size):
		kx, ky = (kernel_size, kernel_size) if type(kernel_size) == int else kernel_size
		sx, sy = img.shape
		s1, s2 = img.strides

		return as_strided(
			img,
			shape=(sx - kx+1, sy - ky+1, kx, ky),
			strides=(s1, s2, s1, s2),
			writeable=False,
		)

	def getSimilar(self, image, threshold=10):
		if self.image_memory_line.data is None:
			return [], []

		# the deviations
		dev = np.array([self.compare(image, img, index) for index, img in enumerate(self.image_memory_line.data)], dtype=np.float64)
		udev = np.unique(dev)[::-1]

		if type(threshold) == float:
			base = threshold

		elif threshold == -1:
			base = 0

		else:
			if threshold > len(udev)-1:
				threshold = len(udev) - 1

			# the base of similarity
			base = udev[threshold-1]

		# the similar images
		similarity_indices = dev.argsort()[::-1]
		similarity_ratios = dev[similarity_indices]
		return similarity_indices[similarity_ratios >= base], similarity_ratios[similarity_ratios >= base]

	def run(self, image, verbose=0):
		'''
		run the instance of an image proc iteration
		'''
		#pick kernel size
		kernel_size = min(image.shape)

		# the segments discovered
		num_of_segments = 0

		# the time of image registration
		timestamp = time.time()

		for low, high in self.discoverSegments(image, kernel_size):
			# get the ones in range of low-high class discovered
			img = ((image >= low) & (image <= high)).astype(np.int8)

			# get the sub segments in image class
			ret, labels = cv2.connectedComponents(img)

			# for each sub segment label
			for label in range(1, ret):
				# avoid low res pixel points
				if image[np.where(labels == label)].std() < 50:
					continue

				# get all positions active(i.e equal to 1)
				pos = np.where(labels == label)

				# all activated pos in image label
				ar1, ar2 = pos

				x1, x2 = min(ar1), max(ar1)+1
				y1, y2 = min(ar2), max(ar2)+1

				# avoid small dimension segments
				if x2-x1 < 3 and y2-y1 < 3:
					continue

				# centroid of segment
				# point = (ar2.mean(), ar1.mean())

				# null space as -1 instead of 0 to avoid confusion
				img_objx = np.full(img.shape, -1, dtype=np.int64)
				img_objx[np.where(labels == label)] = image[np.where(labels == label)]  # crop segment

				# extract segment
				img_objx = img_objx[x1:x2, y1:y2]

				# x-xsize, y-ysize
				xsize, ysize = img_objx.shape

				# save image in standard memory size
				img_obj = np.full((30, 30), -1, dtype=np.int64)

				# save in memory standard image
				img_obj[0:xsize, 0:ysize] = img_objx

				# update the num of segments
				num_of_segments += 1

				# get the similar objects
				similar_images_indices, similarity_ratios = self.getSimilar(img_obj, 5)

				# the image id the segment will take when saved
				image_id = 0 if self.image_memory_line.data is None else len(self.image_memory_line.data)

				self.log(f'\n{label:5d} of {ret-1:5d} segment(s) => {num_of_segments:5d} total segment(s) found')

				# commit current discovery to the cortex
				self.cortex.commitImageInfo(
					image_id,
					img_obj,
					similar_images_indices,
					similarity_ratios,
					timestamp
				)

				# self.image_memory_line.add(image_name, resultant(image))
				self.image_memory_line.add(img_obj, timestamp, allow_duplicate=True)

		# after all segment has been commited
		self.cortex.pushImageProcess()

		return

	def compare(self, img_11, img_22, indexx):
		# get the actual image embedded in negative space
		def getSubImage(img):
			pos = np.where(img != -1)
			ar1, ar2 = pos

			x1, x2 = min(ar1), max(ar1)+1
			y1, y2 = min(ar2), max(ar2)+1

			return img[x1:x2, y1:y2].copy()

		# get the sub image of images
		img_1 = getSubImage(img_11)
		img_2 = getSubImage(img_22)

		# get the shape of the images
		c11, c12 = img_1.shape
		c21, c22 = img_2.shape

		# get the min of the x and size
		r = c21 if c21 < c11 else c11
		c = c22 if c22 < c12 else c12
		
		# size ratio (quantifies reduction/loss of data)
		scale_error = ((r*c)**2) / ((c11 * c12) * (c21 * c22))
	
		# the place holder pixel, using 255 - x cos of background and foreground can blend
		r1 = 255 - img_1[img_1 != -1].mean()
		img_r1 = None if r1 in img_1 else r1

		r2 = 255 - img_2[img_2 != -1].mean()
		img_r2 = None if r2 in img_2 else r2

		# remove -1 cos of interpolation
		if img_r1 is not None:
			img_1[img_1 == -1] = img_r1

		if img_r2 is not None:
			img_2[img_2 == -1] = img_r2

		# resize images (scale it down to minimum)
		img_1 = cv2.resize(img_1.astype(np.uint8), (c, r), interpolation=cv2.INTER_AREA).astype(np.int64)
		img_2 = cv2.resize(img_2.astype(np.uint8), (c, r), interpolation=cv2.INTER_AREA).astype(np.int64)

		# put back -1 after interpolation
		if img_r1 is not None:
			img_1[img_1 == int(img_r1)] = 0

		if img_r2 is not None:
			img_2[img_2 == int(img_r2)] = 255
				
		initial_kernel_size = np.array([r, c])
		delta = initial_kernel_size.min() + 1
		similarities = []
		
		for kernel_size in range(3, delta):
			if kernel_size != 3:
				break

			img1_kernels = np.array(self.getKernels(img_1, kernel_size))
			img2_kernels = np.array(self.getKernels(img_2, kernel_size))

			kernel_similarities = []
			kr, kc, *_ = img1_kernels.shape

			for i in range(kr):
				for j in range(kc):
					kernel_similarities.append(np.diag(np.round(cosine_similarity(img1_kernels[i, j], img2_kernels[i, j]), 4)).mean())
			
			similarities.append(np.array(kernel_similarities).mean())
			
		cos_similarity = np.array(similarities).mean()
		similarity = scale_error * cos_similarity

		# print(f"index = {indexx:2d}, sim = {similarity:.4f}, cos_sim = {cos_similarity:.4f} scale_error = {scale_error:.4f}")
		return cos_similarity
