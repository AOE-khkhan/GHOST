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

def getMnistData():
	with open('test/mnist_data/mnist_data_index.json') as f:
		mnist_data = json.load(f)
		for image_name in mnist_data:
			image_path = '{}/{}.jpg'.format('test/mnist_data', image_name)
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

def getTrainingData(typ):
	# get hand written digit data
	if typ == 'mnist':
		return getMnistData()

	#to dest more definately with fake sharp images
	elif typ == 'fake':
		return simulate_sight()

	else:
		return simulate_experience('test/{}'.format(typ))

def main():
	# initializations
	console = Console()	#console for logging
	timer = Timer()	#timer for timing
	depth = 12 #depth of processors, thought

	# the image image_processor
	img_processor = [ImageProcessor(refid=i, kernel_size=3) for i in range(depth)]

	# get training data
	images = getTrainingData('mnist')
	data = {} #to hold all the data trained with

	c, limit = -1, 30	#counter

	ct = {str(i):0 for i in range(10)}
	for i, (image_name, image, metadata) in enumerate(images):
		
		if ct[metadata] == limit/10:
			continue

		ct[metadata] += 1
		c += 1
		
		if c == limit:
			break

		# get the grey version of image
		image = toGrey(image, 1/3, 1/3, 1/3)

		# the dimensions
		if len(image.shape) == 3:
			w, h, d = image.shape
		
		else:
			(w, h), d = image.shape, 1

		# all collected data
		data[image_name] = metadata
		
		# report image information
		console.log('LOADING IMAGE_{}: {:>7}, initial dimension: width = {}, height = {}, depth = {}, metadata: {}'.format(c+1, image_name, w, h, d, metadata))

		# if process
		verbose = 1 if i == -1 or 1 else 0

		# run against timer
		def image_processor_getSimilar():
			similar, similar_ratio = img_processor[processor_index].getSimilar(image, verbose=verbose)
			console.log(' SIMILAR IMAGES: {}'.format('None' if len(similar) == 0 else ''))
			for xi, x in enumerate(similar):
				console.log('  {:<10s}({:<0.4f}): {}'.format(x, similar_ratio[xi], data[x]))
				xx = '{}({})'.format(x, data[x])
				if xx not in sim:
					sim[xx] = similar_ratio[xi]

				else:
					sim[xx] += similar_ratio[xi]

			return similar, similar_ratio

		# run against timer
		def image_processor_register():
			feature_map = img_processor[processor_index].register(image, image_name, verbose=verbose)
			if verbose:
				# ref path
				fmap_path = 'results/feature_maps/{}/{}.txt'.format(processor_index, image_name)
				img_path = 'results/images/{}/{}.txt'.format(processor_index, image_name)
				
				# validate the folders
				validateFolderPath(os.path.dirname(fmap_path))
				validateFolderPath(os.path.dirname(img_path))
				
				# save the maps				
				np.savetxt(fmap_path, feature_map)
				np.savetxt(img_path, image)

			return feature_map

		sim = {}
		for processor_index in range(depth):
			console.log('PROCESSOR {}: --------->'.format(processor_index+1))
			similar_images = timer.run(image_processor_getSimilar)
			image = timer.run(image_processor_register)

		sim = sorted(sim.items(), key=lambda x: x[1], reverse=True)
		for sm, smr in sim:
			print('{:<10s}: {:<0.4f}'.format(sm, smr))

		console.log('-'*100, '\n')
		

	for processor_index in range(depth):
		console.log('PROCESSOR {}: --------->'.format(processor_index+1))
		if img_processor[processor_index].image_memory_line.indices is not None:
			# np.savetxt('indices.txt', img_processor[i].kernel_memory_line.indices)
			console.log(' {} data loaded into memory\n {} kernel(s) extracted\n {} cluster(s) discoverd'.format(
				len(img_processor[processor_index].image_memory_line.indices),
				len(img_processor[processor_index].kernel_memory_line.indices),
				len(img_processor[processor_index].kernel_memory_line.clusters)
				)
			)

	with open('train.json', 'w') as f:
		json.dump(data, f)

if __name__ == '__main__':
	main()