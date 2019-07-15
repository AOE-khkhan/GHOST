# from python std lib
import os

# from third party
import cv2

# from the lib code
from image_processor import ImageProcessor
from console import Console
from timer import Timer
from utils import load_image

def simulate_experience(img_dir=None):
	# the image paths
	image_names = os.listdir(img_dir)

	for image_name in image_names:
		yield image_name

def main():
	# initializations
	console = Console()	#console for logging
	timer = Timer()	#timer for timing

	# the image image_processor
	img_processor = ImageProcessor(refid=3)

	# the data to simulate experience
	img_dir = 'test/images'
	image_names = simulate_experience(img_dir)

	c = 0	#counter

	for image_name in image_names:
		# load the images
		image = load_image('{}/{}'.format(img_dir, image_name))

		w, h, d = image.shape
		console.log('loading image: {:>11}, initial dimension: width = {}, height = {}, depth = {}'.format(image_name, w, h, d))

		# run against timer
		def image_processor_register():
			img_processor.register(image)

		result = timer.run(image_processor_register)

		c += 1
		if c == 2:
			break

if __name__ == '__main__':
	main()