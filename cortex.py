class Cortex(object):
	''' central processing unit of GHOST'''
	def __init__(self):
		# the image processor
		self.image_processor = None
	
	def commitImageInfo(self, info):
		# the info collected
		image_id, point, (similar_images, similarity_ratios) = info

		# for i, similar_image in enumerate(similar_images):
		# 	print(similar_image, similarity_ratios[i])
			# self.image_processor.image_memory_line.experiences[similar_image]

	def commitKeyInfo(self, info):
		return

		

