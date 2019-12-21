#import from third party lib
import pandas as pd

from collections import Counter, defaultdict

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
		self.ik_co_occurrence_probability = defaultdict(float)

		# the output prediction from cortex
		self.result_vector = None

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

		keyboard_context_labels = set()

		for similar_image_index, similarity_ratio in zip(similar_images_indices, similarity_ratios):
			# find the related meatadata to similar image
			similar_image_metadata = self.getImageMetadata(similar_image_index)

			# verify image on metadata model
			if similar_image_metadata in keyboard_context_labels:
				pass

			# add to keyboard context
			keyboard_context_labels.add(similar_image_metadata)

			# add to the probability
			self.ik_co_occurrence_probability[similar_image_metadata] += similarity_ratio

			print(f'image: {image_index:3d}[{image_metadata}] => {similar_image_index:3d}[{similar_image_metadata}], ')

		return

	def commitKeyInfo(self, info):
		return

	def pushImageProcess(self, number_of_inference=1):
		# output the results of image process
		ik_co_occurrence_probability = pd.DataFrame(self.ik_co_occurrence_probability, index=['freq']).T.sort_values('freq', ascending=False)
		
		# normalize by mean
		ik_co_occurrence_probability /= number_of_inference
		
		# set the result_vector
		self.result_vector = ik_co_occurrence_probability.rename(columns={'freq':'probability'})
		
		# show the final results
		self.log(f'\nResult\n==========\n{self.result_vector}\n')

		# re-initialize the image-keyboard co_occurence overall probaility
		self.ik_co_occurrence_probability = defaultdict(float)

		return 

	def getImageMetadata(self, image_index, inc=1):
		'''
			returns the [keyboard]metadata associted to the time period[inc] of image
		'''
		if type(image_index) == float:
			image_timestamp = image_index

		else:
			image_timestamp = self.image_processor.image_memory_line.metadata[image_index][TIMESTAMP]

		low = image_timestamp - inc # the lower end of simulteniety
		
		# the metadata to serach from
		df = self.keyboard_processor.keyboard_memory_line.metadata.T

		# find the related meatadata to image
		image_metadata = df[(df[TIMESTAMP] > low) & (df[TIMESTAMP] <= image_timestamp)][TIMESTAMP]

		if len(image_metadata) and not pd.isnull(image_metadata.idxmax()):
			return self.keyboard_processor.keyboard_memory_line.data[image_metadata.idxmax()]

		return
		

