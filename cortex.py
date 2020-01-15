# from python std lib
from collections import Counter, defaultdict

# import from third party lib
import pandas as pd
import numpy as np

# import lib code
from context import Context
from console import Console
from memory_line import MemoryLine

from utils import TIMESTAMP, index_row_in_array


class Cortex(object):
	''' central processing unit of GHOST'''

	def __init__(self):
		# the image processor
		self.image_processor = None
		self.keyboard_processor = None

		# set up memory line
		self.experience_memory_line = MemoryLine(has_experience=False)

		# initialize console
		self.console = Console()
		self.log = self.console.log
		self.console.setLogState(True)

		# the history
		context_maxlength = 10
		self.context = Context(context_maxlength)

		# the overall ratio of image relation with metadata
		self.image_similar_images = defaultdict(list)
		self.image_similar_images_sources = defaultdict(list)

		# the output prediction from cortex
		self.result = [], 0

	def trustFactor(self, size):
		'''
		return a value in range [0-1] quantifying trust factor based on data size
			transforms integer to float in range 0-1

		'''

		if type(size) == list:
			size = len(size)

		return size / (size + 1)

	def commitImageInfo(self, image_index, image, similar_images_indices, similarity_ratios, timestamp):
		if self.keyboard_processor.keyboard_memory_line.data is None:
			return

		# find the related meatadata to image
		image_metadata = self.getImageMetadata(timestamp)

		for similar_image_index, similarity_ratio in zip(similar_images_indices, similarity_ratios):
			
			# find the related meatadata to similar image
			similar_image_metadata = self.getImageMetadata(similar_image_index)
			
			# add to the probability
			self.image_similar_images[similar_image_index].append(similarity_ratio)
			self.image_similar_images_sources[similar_image_index].append(image_index)

			print(
				f'image: {image_index:3d}[{image_metadata}] => {similar_image_index:3d}[{similar_image_metadata}], ',
				f'=> {similarity_ratio:.4f}',
			)
			
		return

	def commitKeyInfo(self, info):
		return

	def pushImageProcess(self):
		# find the mean similarities of image
		image_similar_images = {
			similar_image_index:np.array(values).mean() for similar_image_index, values in self.image_similar_images.items()
		}

		# output the results of image process
		image_similar_images = pd.DataFrame(image_similar_images, index=['similarity']).T.sort_values('similarity', ascending=False)

		# find the most similar image and its probability
		similarities = image_similar_images['similarity'] #the similarity ratios
		max_similarity = similarities.max() #max similarity ratio

		# if max_similarity <= 0.5:
		# 	return

		# most similar images and their metadata
		most_similar_images_indices = similarities[similarities == max_similarity].index

		# the metadata of the most similar images
		most_similar_images_metadata = [self.getImageMetadata(most_similar_image_index) for most_similar_image_index in most_similar_images_indices]

		# set the result_vector
		self.result = most_similar_images_metadata, max_similarity

		# show the final results
		self.log(f'\nResult\n==========\n{image_similar_images}\n{self.result}\n')

		# re-initialize the image-keyboard co_occurence overall probaility
		self.image_similar_images = defaultdict(list)
		self.image_similar_images_sources = defaultdict(list)

		return

	def getImageMetadata(self, image_index, inc=1):
		'''
			returns the [keyboard]metadata associted to the time period[inc] of image
		'''
		if type(image_index) == float:
			image_timestamp = image_index

		else:
			image_timestamp = self.image_processor.image_memory_line.metadata[
				image_index][TIMESTAMP]

		low = image_timestamp - inc  # the lower end of simulteniety

		# the metadata to serach from
		df = self.keyboard_processor.keyboard_memory_line.metadata.T

		# find the related meatadata to image
		image_metadata = df[(df[TIMESTAMP] > low) & (
			df[TIMESTAMP] <= image_timestamp)][TIMESTAMP]

		if len(image_metadata) and not pd.isnull(image_metadata.idxmax()):
			return self.keyboard_processor.keyboard_memory_line.data[image_metadata.idxmax()]

		return
