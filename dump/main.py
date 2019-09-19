# from python std lib
import os, random, time, json

# from third party
import cv2
import numpy as np

# from the lib code
from image_processor import ImageProcessor
from console import Console
from timer import Timer
from utils import load_image, toGrey, validateFolderPath

def simulate_experience(img_dir):
	# the image paths
	image_names = os.listdir(img_dir)
	# random.shuffle(image_names)

	for image_name in image_names:
		yield image_name, load_image('{}/{}'.format(img_dir, image_name))[:28, :28], 'metadata'

def getDataset(dataset_name, ext):
	# with open('test/img/labels.json') as f:
	with open(f'test/{dataset_name}/labels.json') as f:
		mnist_data = json.load(f)
		for image_name in mnist_data:
			image_path = f'test/{dataset_name}/{image_name}.{ext}'
			yield image_name, load_image(image_path),  mnist_data[image_name]

def simulate_sight():
	# the image paths
	image_template = np.zeros((28, 28, 3))

	start_i = 0
	for i in range(1, (28//7)+1):
		end_i = i*7
		
		start_j = 0
		for j in range(1, (28//7)+1):
			end_j = j*7
			image = image_template.copy()
			image[start_i:end_i, start_j:end_j] = 255
			
			start_j = end_j
			name = str(time.time()).replace(".", "_")
			yield 'nameless_{:0<20}.jpg'.format(name), image, 'metadata'

		start_i = end_i

def getTrainingData(typ, ext='jpg'):
	#to dest more definately with fake sharp images
	# elif typ == 'fake':
	# 	return simulate_sight()

	# else:
	# 	return simulate_experience('test/{}'.format(typ))
	
	return getDataset(typ, ext)

def main():
	# initializations
	console = Console()	#console for logging
	timer = Timer()	#timer for timing
	depth = 1 #depth of processors, thought

	# the image image_processor
	img_processor = [ImageProcessor(refid=i, kernel_size=3) for i in range(depth)]

	# get training data
	images = getTrainingData('plate_numbers', 'jpeg')
	data = {} #to hold all the data trained with

	c, limit = -1, 60	#counter

	ct = {str(i if i <= 9 else chr(65-10+i)):0 for i in range(36)}
	for i, (image_name, image, metadata) in enumerate(images):
		if ct[metadata] == limit/10:
			continue

		c += 1	
		if c == limit:
			break

		# get the grey version of image
		image = toGrey(image)

		# the dimensions
		if len(image.shape) == 3:
			w, h, d = image.shape
		
		else:
			(w, h), d = image.shape, 1
		
		# report image information
		console.log('LOADING IMAGE_{}: {:>7}, initial dimension: width = {}, height = {}, depth = {}, metadata: {}'.format(c+1, image_name, w, h, d, metadata))

		# if process
		verbose = 1 if i == -1 or 1 else 0

		# run against timer
		def image_processor_run():
			process = img_processor[processor_index].run(image, image_name, verbose=verbose)
			similar, similar_ratio = [], []
			
			rstate = False
			data[image_name] = metadata
			console.log(' SIMILAR IMAGES:')

			for similar, similar_ratio in process:
				rstate = True
				for xi, x in enumerate(similar):
					console.log('  {:<10s}({:<0.4f}): {}'.format(x, similar_ratio[xi], data[x]))
					xx = '{}({})'.format(x, data[x])
					if xx not in sim:
						sim[xx] = similar_ratio[xi]

					else:
						sim[xx] += similar_ratio[xi]
				print()

			if not rstate:
				# console.log(' SIMILAR IMAGES: None')
				return similar, similar_ratio

			# all collected data
			ct[metadata] += 1

			return similar, similar_ratio

		sim = {}
		for processor_index in range(depth):
			console.log('PROCESSOR {}: --------->'.format(processor_index+1))
			similar_images, similarity_ratio = timer.run(image_processor_run)

		sim = sorted(sim.items(), key=lambda x: x[1], reverse=True)
		for sm, smr in sim:
			print('{:<10s}: {:<0.4f}'.format(sm, smr))

		console.log('-'*100, '\n')
		

	for processor_index in range(depth):
		console.log('PROCESSOR {}: --------->'.format(processor_index+1))
		if img_processor[processor_index].image_memory_line.indices is not None:
			# np.savetxt('indices.txt', img_processor[i].kernel_memory_line.indices)
			console.log(' {} data loaded into memory\n'.format(len(img_processor[processor_index].image_memory_line.indices)))
		
	for metadata in ct:
		print('{}: {}'.format(metadata, ct[metadata]))

	with open('train.json', 'w') as f:
		json.dump(data, f)

if __name__ == '__main__':
	main()
