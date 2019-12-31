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

		# batch of images to train the network with
		self.train_images_input_batch = []
		self.train_images_output_batch = []

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

		#similar_images_indices set
		set_of_similar_images_indices = set(similar_images_indices)

		for similar_image_index, similarity_ratio in zip(similar_images_indices, similarity_ratios):
			
			# find the related meatadata to similar image
			similar_image_metadata = self.getImageMetadata(similar_image_index)

			# get the similar objects of similar objects
			ssimilar_images_indices, ssimilarity_ratios = self.image_processor.getSimilar(
				self.image_processor.image_memory_line.data[similar_image_index], 3
			)

			#similar_images_indices set
			sset_of_similar_images_indices = set(ssimilar_images_indices)

			# common in similar_images_indices and ssimilar_images_indices
			common_indices = set_of_similar_images_indices.intersection(sset_of_similar_images_indices)

			# similarities of the common similarities
			common_ratio = []

			for ssimilar_image_index, ssimilarity_ratio in zip(ssimilar_images_indices, ssimilarity_ratios):

				if ssimilar_image_index not in common_indices:
					continue

				common_ratio.append(ssimilarity_ratio)

				# find the related meatadata to similar image
				ssimilar_image_metadata = self.getImageMetadata(ssimilar_image_index)

				print(
					f'  image: {similar_image_index:3d}[{similar_image_metadata}] => {ssimilar_image_index:3d}[{ssimilar_image_metadata}], ',
					f'=> {ssimilarity_ratio:.4f}',
				)

			common_ratio = np.array(common_ratio).sum()**2
			common_ratio /= (len(set_of_similar_images_indices) * len(sset_of_similar_images_indices))

			# add to the probability
			self.image_similar_images[similar_image_index].append(similarity_ratio)
			self.image_similar_images_sources[similar_image_index].append(image_index)

			print(
				f'image: {image_index:3d}[{image_metadata}] => {similar_image_index:3d}[{similar_image_metadata}], ',
				f'=> {similarity_ratio:.4f}, intersection = {common_ratio:.4f} -> {similarity_ratio * common_ratio:.4f}',
			)
			
		return

	def commitKeyInfo(self, info):
		return

	def pushImageProcess(self):
		# find the mean similarities of image
		image_similar_images = {
			similar_image_index:np.array(values).mean()*self.trustFactor(values) for similar_image_index, values in self.image_similar_images.items()
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
		most_similar_images_metadata = []
		
		# update the batch of images
		for most_similar_image_index in most_similar_images_indices:

			# save most similar metadata
			most_similar_images_metadata.append(self.getImageMetadata(most_similar_image_index))

			# similarities between sim image and image
			sims = self.image_similar_images[most_similar_image_index]

			# get all similar pairs
			for i, image_index in enumerate(self.image_similar_images_sources[most_similar_image_index]):
				
				# similarity betweenn most_similar_image_index and image index
				sim = sims[i]

				# if sim  == max_similarity:
				# 	continue

				# save a similar pair
				self.train_images_input_batch.append([
					self.image_processor.image_memory_line.data[image_index],
					self.image_processor.image_memory_line.data[most_similar_image_index],
				])

				actual_prediction = int(most_similar_images_metadata[-1] != self.getImageMetadata(image_index))
				self.train_images_output_batch.append([[0], [actual_prediction]])

		# set the result_vector
		self.result = most_similar_images_metadata, max_similarity

		# show the final results
		self.log(
			f'\nResult\n==========\n{image_similar_images}\n{self.result}\n')

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
