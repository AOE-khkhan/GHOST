# from python std lib
import os, random, time, json
from threading import Thread

# from third party
import cv2, pyautogui
import numpy as np

# from the lib code
from image_processor import ImageProcessor
from cortex import Cortex
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
				# cv2.imwrite(f'results/{image_name}.jpg', game_object.gw)
				yield image_name, game_object.gw, move
		
		break


def main():
	# initializations
	console = Console()	#console for logging
	timer = Timer()	#timer for timing

	# the image image_processor
	cortex = Cortex()
	image_processor = ImageProcessor(cortex=cortex, kernel_size=3)

	# get training data
	images = getSimulateGamePlay()
	c, limit = 1, 60  # to hold all the data trained with

	for i, (image_name, image, metadata) in enumerate(images):
		c += 1	
		if c == limit:
			break
		

		# the dimensions
		if len(image.shape) == 3:
			w, h, d = image.shape
		
		else:
			(w, h), d = image.shape, 1
		
		# report image information
		console.log('LOADING IMAGE_{}: {:>7}, initial dimension: width = {}, height = {}, depth = {}, metadata: {}'.format(c+1, image_name, w, h, d, metadata))

		# processor processing results report
		timer.run(lambda: image_processor.run(image, image_name, verbose=1))

		console.log('-'*100, '\n')
		
	# processor stats report
	if image_processor.image_memory_line.indices is not None:
		console.log(' {} data loaded into memory\n'.format(len(image_processor.image_memory_line.indices)))


if __name__ == '__main__':
	main()
