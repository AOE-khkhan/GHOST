# from python std lib
import os

# from third party
import cv2
import numpy as np

# from the lib code
from image_processor import ImageProcessor
from console import Console
from timer import Timer
from utils import load_image

def simulate_experience(img_dir=None):
	# the image paths
	image_names = sorted(os.listdir(img_dir))

	for image_name in image_names:
		yield image_name

def main():
	# initializations
	console = Console()	#console for logging
	timer = Timer()	#timer for timing

	# the image image_processor
	img_processor = ImageProcessor(refid=3)

	# the data to simulate experience
	img_dir = 'test/images2'
	image_names = simulate_experience(img_dir)

	c = 0	#counter

	for i, image_name in enumerate(image_names):
		# load the images
		image = load_image('{}/{}'.format(img_dir, image_name))[:28, :28]

		w, h, d = image.shape
		console.log('loading image: {:>11}, initial dimension: width = {}, height = {}, depth = {}'.format(image_name, w, h, d))

		verbose = 1 if i > -1 else 0

		# run against timer
		def image_processor_getSimilar():
			img_processor.getSimilar(image)

		# run against timer
		def image_processor_register():
			img_processor.register(image, idx=image_name.split('.')[0], verbose=verbose)

		similar_images = timer.run(image_processor_getSimilar)
		feature_map = timer.run(image_processor_register)

		print()
		c += 1
		if c == -2:
			break
	
	# np.savetxt('indices.txt', img_processor.kernel_memory_line.indices)

if __name__ == '__main__':
	main()