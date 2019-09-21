# from python std lib
import os, random, time, json
from threading import Thread

# from third party
import cv2, pyautogui
import numpy as np

# from the lib code
from image_processor import ImageProcessor
from console import Console
from timer import Timer
from utils import load_image, toGrey, validateFolderPath

from game import Game, UP, DOWN, LEFT, RIGHT, NONE

# the key codes for moves
# MOVES = {UP: 'w', DOWN: 'z', LEFT: 'a', RIGHT: 's'}
MOVES = {UP: 119, DOWN: 122, LEFT: 97, RIGHT: 115}

def getSimulateGamePlay(train_game_play_path='data/game_plays/game_play.json'):
	# the start and end grid
	game_play_path = 'game_play.json'

	with open(train_game_play_path) as f:
		game_plays = json.load(f)

	for game_configuration in game_plays:
		a, b, c, d = list(map(int, game_configuration.split(' ')))

		# the start and the target
		start, target = (a, b), (c, d)
	
		# initialize game
		game_object = Game(start=start, target=target, game_play_path=game_play_path)

		#run discretely
		game_object.runDiscretely()

		# run continuously
		# t1 = Thread(target=game_object.runContinously, args=());t1.start()
		# pyautogui.hotkey('alt', 'tab')
		# game_object.runContinously()

		for game_play in game_plays[game_configuration]:
			for move in game_play:
				# make a move
				game_object.runDiscretely(MOVES[move])

				image_name = time.time();image_name = f'{image_name:.4f}'[-7:]
				cv2.imwrite(f'results/{image_name}.jpg', game_object.gw)
				yield image_name, game_object.gw, move
		
		break


def main():
	# initializations
	console = Console()	#console for logging
	timer = Timer()	#timer for timing
	depth = 1 #depth of processors, thought

	# the image image_processor
	img_processor = [ImageProcessor(refid=i, kernel_size=3) for i in range(depth)]

	# get training data
	images = getSimulateGamePlay()
	c, limit, data, data_c = 1, 60, {}, {}  # to hold all the data trained with

	for i, (image_name, image, metadata) in enumerate(images):
		if metadata not in data_c:
			data_c[metadata] = 0

		c += 1	
		if c == limit:
			break
		
		# image = toGrey(image)

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
			data_c[metadata] += 1

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
		
	for metadata in data_c:
		print('{}: {}'.format(metadata, data_c[metadata]))

	with open('train.json', 'w') as f:
		json.dump(data, f)


if __name__ == '__main__':
	main()
