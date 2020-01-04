# import from std lib
import time
from collections import defaultdict

# import from third party lib
import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.metrics.pairwise import cosine_similarity

# import lib code
from context import Context
from console import Console
from memory_line import MemoryLine

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

	def getSimilar(self, image, threshold=10, verbose=0):
		# if not verbose:
		# 	return

		if self.image_memory_line.data is None:
			return [], []

		# the deviations
		dev = np.array([self.compare(image, img, index) for index, img in enumerate(self.image_memory_line.data)], dtype=np.float64)
		indices = np.array(range(len(dev)))

		for i in range(4):
			selected = np.where(dev[:, i] == dev[:, i].min())
			dev = dev[selected]
			indices = indices[selected]

			print(dev.shape)

			if len(dev == 1):
				return indices, dev[:, -1]

		
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

	def compare1(self, img_11, img_22):
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

		# base area of image (scaled down vrsion)
		base_area = r * c

		# size ratio (quantifies reduction/loss of data)
		sr1 = base_area / (img_1.shape[0] * img_1.shape[1])
		sr2 = base_area / (img_2.shape[0] * img_2.shape[1])

		# the trust factor for scaling down (consequence of scaling down)
		scale_weight = sr1*sr2

		img1_before_before = img_1.copy()
		img2_before_before = img_2.copy()

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
		img_1 = cv2.resize(img_1.astype(np.uint8), (c, r), interpolation=cv2.INTER_AREA).astype(np.float64)
		img_2 = cv2.resize(img_2.astype(np.uint8), (c, r), interpolation=cv2.INTER_AREA).astype(np.float64)

		img1_before = img_1.copy()
		img2_before = img_2.copy()

		# put back -1 after interpolation
		if img_r1 is not None:
			img_1[img_1 == int(img_r1)] = -1

		if img_r2 is not None:
			img_2[img_2 == int(img_r2)] = -1

		# find the centroid of image 1 and 2
		img_1a, img_1b = np.where(img_1 != -1)
		try:
			img_1a, img_1b = int(img_1a.mean()), int(img_1b.mean())

		except Exception as e:
			print(img_r1, r1)
			print(img1_before_before)
			print(img1_before)
			print(img_1)
			print('image 1:', e)
			return 0

		img_2a, img_2b = np.where(img_2 != -1)
		try:
			img_2a, img_2b = int(img_2a.mean()), int(img_2b.mean())

		except Exception as e:
			print(img_r2, r2)
			print(img2_before_before)
			print(img2_before)
			print(img_2)
			print('image 2:', e)
			return 0

		# overllap the centroids
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

		# get the base image canvas to compare as
		r = max((img_1x2, img_2x2))
		c = max((img_1y2, img_2y2))
		
		# the new images
		img_n1, img_n2 = np.full((r, c), -1, dtype=np.float64), np.full((r, c), -1, np.float64)

		# the new images as size of the overlap
		img_n1[img_1x1:img_1x2, img_1y1:img_1y2] = img_1
		img_n2[img_2x1:img_2x2, img_2y1:img_2y2] = img_2

		# get the base similar image
		img_b = img_n1.copy()
		img_b[img_n2 == -1] = -1

		m = 255 if np.amax(img_n1) > 1 or np.amax(img_n2) > 1 else 1
		m = 255

		img_n1[img_n1 == -1] = 255 - img_n2[img_n2 != -1].mean()
		img_b[img_b == -1] = 255 - img_n2[img_n2 != -1].mean()
		img_n2[img_n2 == -1] = 255 - img_n1[img_n1 != -1].mean()

		z1 = (m - abs(img_b - img_n2).mean()) / m
		z2 = (m - abs(img_b - img_n1).mean()) / m
		z3 = (m - abs(img_n2 - img_n1).mean()) / m

		# zx = scale_weight * (z1 + z2 + z3) / 3
		# zx = (z1 + z2 + z3) / 3
		# zx = (scale_weight + ((z1 + z2 + z3) / 3)) / 2
		zx = scale_weight * z3

		delta = abs(img_n2 - img_n1)

		delta1 = delta.copy()
		delta1[img_n1 == -1] = 255
		z11 = (255 - delta1.mean()) / 255

		delta2 = delta.copy()
		delta2[img_n2 == -1] = 255
		z22 = (255 - delta2.mean()) / 255

		delta3 = delta.copy()
		delta3[img_n1 == -1] = 255
		delta3[img_n2 == -1] = 255
		z33 = (255 - delta3.mean()) / 255
		
		# zy = scale_weight * ((z11 + z22) / 2) * z33
		# zy = scale_weight * z11 * z22 * z33
		zy = scale_weight * (z11 + z22 + z33) / 3
		# zy = scale_weight * ((.025 * z11) + (.025 * z22) + (.95 * z33))
		# zy = scale_weight * z33
		return zy

	def compare(self, img_11, img_22, indexx):
		def centralize(img_coordinates):
			return np.array([row - row.mean() for row in img_coordinates], dtype=np.float64)

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
		
		# pixel points before scale alteration
		n_pixels_before = np.array([len(np.where(img_1 != -1)[0]), len(np.where(img_2 != -1)[0])])
	
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
		img_1 = cv2.resize(img_1.astype(np.uint8), (c, r), interpolation=cv2.INTER_AREA).astype(np.float64)
		img_2 = cv2.resize(img_2.astype(np.uint8), (c, r), interpolation=cv2.INTER_AREA).astype(np.float64)

		# put back -1 after interpolation
		if img_r1 is not None:
			img_1[img_1 == int(img_r1)] = -1

		if img_r2 is not None:
			img_2[img_2 == int(img_r2)] = -1
		
		# size ratio (quantifies reduction/loss of data)
		n_pixels_after = np.array([len(np.where(img_1 != -1)[0]), len(np.where(img_2 != -1)[0])])
		scale_error = (abs(n_pixels_after - n_pixels_before) / n_pixels_before).mean()
		
		# find the coordinates of image 1 and 2 with center as centroid
		image_1_coordinates = np.array(np.where(img_1 != -1)).T
		image_2_coordinates = np.array(np.where(img_2 != -1)).T
		
		coord1, coord2 = (centralize(image_1_coordinates), centralize(image_2_coordinates))
		imgX, imgY = img_1.copy(), img_2.copy()
		
		if len(image_1_coordinates) > len(image_2_coordinates):
			coord1, coord2 = coord2, coord1
			image_1_coordinates, image_2_coordinates = image_2_coordinates, image_1_coordinates
			imgX, imgY = img_2.copy(), img_1.copy()
				
		deviation_angles, deviation_distances, deviation_pixels = [], [], []
		active_positions = np.array(range(len(coord2)))
		alpha, MAX_SIDE = 10e-10, 28
		
		for index, coord in enumerate(coord1):
			unmatched = coord2[(active_positions, )]
			
			if not len(active_positions):
				break
				
			#position difference, alpha to avoid zero values and round to make same vector have sim =1
			vec_sims = np.round(cosine_similarity([coord+alpha], unmatched+alpha)[0], 4)
		
			if vec_sims.max() < 0:
				continue
			
			coord_sign = coord[0] * coord[1]
			sign = unmatched[:, 0] * unmatched[:, 1]
			# most_sim_indices = active_positions[vec_sims == vec_sims.max()]
	        most_sim_indices = active_positions[((vec_sims == vec_sims.max()) & ((coord_sign > 0) == (sign > 0)))]

			#the distance away for pixel coord
			deviation_distance = abs(coord - coord2[most_sim_indices]).mean(1)
			
			#collect error in angle and distance
			deviation_angles.append(1 - vec_sims.max())
			deviation_distances.append(deviation_distance.min())
			
			#the index of pixel eliminated
			eliminated_index = most_sim_indices[deviation_distance.argmin()]

			#teh error from position and color
			x1, y1 = image_1_coordinates[index]
			x2, y2 = image_2_coordinates[eliminated_index]
			deviation_pixels.append(abs(imgX[x1, y1] - imgY[x2, y2]))
			
			#the diff in position (and pixel)
			active_positions = np.delete(active_positions, np.argwhere(active_positions==eliminated_index))
			
		unmatched = coord2[(active_positions,)]
		unmatched = abs(unmatched).mean(1).mean() if len(unmatched) else 0

		deviation_distance = np.array(deviation_distances).mean() / MAX_SIDE
		deviation_angle = np.array(deviation_angles).mean()
		deviation_pixel = np.array(deviation_pixels).mean()

		f = lambda x: x / (x + 1)
		factor = 1
		# factor = f(len(coord1) / len(coord2))
		# factor = (len(coord1) / len(coord2))
		
		error = ((deviation_angle) + (deviation_distance) + f(f(unmatched)) + f(scale_error)) / 4
		# error = (np.array([deviation_angle, deviation_distance, f(scale_error), f(unmatched)]) * np.array([1/2, 1/2, 0, .0])).sum()
		# error = (error + f(f(unmatched))) / 2

		print(f"index = {indexx:2d}, angle_error = {deviation_angle:.4f}, distance_error = {deviation_distance:.4f}, scale_error = {f(scale_error):.4f}, unmatched = {f(f(unmatched)):.4f} = > {1 - error:.4f}")
		# return factor * (1-error)
		return np.array([deviation_angle, deviation_distance, f(scale_error), f(unmatched), 1-error])
