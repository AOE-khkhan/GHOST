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

	def commitImageInfo(self, info):
		# the info collected
		image_index, point, (similar_images, similarity_ratios) = info

		# the data in memory
		image_data = self.image_processor.image_memory_line.data

		# the data in keyboard memory
		keyboard_data = self.keyboard_processor.keyboard_memory_line.data

		# find the related meatadata to image
		image_metadata = self.getImageMetadata(image_index)

		num_of_similar = len(similar_images)

		for similar_image_index, similarity_ratio in zip(similar_images, similarity_ratios):
			# find the related meatadata to similar image
			similar_image_metadata = self.getImageMetadata(similar_image_index)
			
			if similar_image_metadata is not None:
				# the indices(apperance) of segment in memory
				indices = index_row_in_array(image_data[similar_image_index], image_data)
				
				# all metadata associated to a property(segment) of image
				metadatas = [self.getImageMetadata(index) for index in indices]

			else:
				metadatas = [None]

			# metadata co_occurrence freq distribution
			freq_distribution = Counter(metadatas)

			# the number of apperance of image(segment)
			num_of_occurrence = len(metadatas)

			# the quantity-quality factor
			factor = self.trustFactor(num_of_occurrence)

			for metadata, freq in freq_distribution.items():
				# the indices(apperance) of metadata in memory
				metadata_indices = index_row_in_array(metadata, keyboard_data)

				num_of_metadata_occurrence = len(metadata_indices)

				metadata_factor = self.trustFactor(num_of_metadata_occurrence)

				self.ik_co_occurrence_probability[metadata] += (
					factor * metadata_factor * similarity_ratio * freq * freq) / (
						num_of_occurrence * num_of_metadata_occurrence * num_of_similar
				)

			print(
				f'image: {image_index:3d}[{image_metadata}] => {similar_image_index:3d}[{similar_image_metadata}], ' +
			 	f'similarity_ratio = {similarity_ratio:7.4f}, centroid = ({point[0]:7.4f}, {point[1]:7.4f}) {metadatas}'
			)
		return

	def commitKeyInfo(self, info):
		return

	def pushImageProcess(self, number_of_inference=1):
		# output the results of image process
		ik_co_occurrence_probability = pd.DataFrame(self.ik_co_occurrence_probability, index=['freq']).T.sort_values('freq', ascending=False)
		ik_co_occurrence_probability /= number_of_inference
		# ik_co_occurrence_probability /= ik_co_occurrence_probability.sum().sum()
		
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

		image_timestamp = self.image_processor.image_memory_line.metadata[image_index][TIMESTAMP]
		low = image_timestamp - inc # the lower end of simulteniety
		
		# the metadata to serach from
		df = self.keyboard_processor.keyboard_memory_line.metadata.T

		# find the related meatadata to image
		image_metadata = df[(df[TIMESTAMP] > low) & (df[TIMESTAMP] <= image_timestamp)][TIMESTAMP]

		if len(image_metadata) and not pd.isnull(image_metadata.idxmax()):
			return self.keyboard_processor.keyboard_memory_line.data[image_metadata.idxmax()]

		return
		

