# from python std lib
import os

# from the lib code
from image_processor import ImageProcessor
from console import Console

def main():
	# initializations
	#console for logging
	console = Console()

	# the image image_processor
	size = 24
	img_processor = ImageProcessor(size=size)

	# memory strip
	# mms = MagneticMemoryStrip()

	# test the MagneticMemoryStrip
	# mms.runTests()

	img_dir = 'test/images'

	# the image paths
	image_names = os.listdir(img_dir)
	
	for image_name in image_names:
		# load the images
		image = img_processor.load_image('{}/{}'.format(img_dir, image_name))

		w, h, d = image.shape
		console.log('loading image: {:>11}, initial dimension: width = {}, height = {}, depth = {}'.format(image_name, w, h, d))

		img_processor.register(image)
	

if __name__ == '__main__':
	main()