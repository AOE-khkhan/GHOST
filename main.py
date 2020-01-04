# from python std lib
import os
import random
import time
import json
from threading import Thread

# from third party
import cv2
import numpy as np
import pandas as pd

from keras.datasets import mnist

# from the lib code
from image_processor import ImageProcessor
from keyboard_processor import KeyboardProcessor
from cortex import Cortex
from console import Console
from timer import Timer
from utils import load_image, toGrey, validateFolderPath

from game import Game, UP, DOWN, LEFT, RIGHT, NONE

# the key codes for moves
# MOVES = {UP: 'w', DOWN: 'z', LEFT: 'a', RIGHT: 's'}
MOVES = {UP: 119, DOWN: 122, LEFT: 97, RIGHT: 115}

# the sensors
IMAGE_SENSOR, KEYBOARD_SENSOR = '__image_sensor__', '__keyboard_sensor__'
SENSORS = [IMAGE_SENSOR, KEYBOARD_SENSOR]


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
        game_object = Game(start=start, target=target,
                           game_play_path=game_play_path)

        # run discretely
        game_object.runDiscretely()

        # run continuously
        # t1 = Thread(target=game_object.runContinously, args=());t1.start()
        # pyautogui.hotkey('alt', 'tab')
        # game_object.runContinously()

        for game_play in game_plays[game_configuration]:
            for move in game_play:
                # make a move
                game_object.runDiscretely(MOVES[move])

                image_name = time.time()
                image_name = f'{image_name:.4f}'[-7:]
                # cv2.imwrite(f'results/{image_name}.jpg', game_object.gw)
                yield image_name, game_object.gw.astype(np.int64), move

        break


def simulateMNISTDigitRecognition(n=None):
    # (x_train, y_train), (x_test, y_test) = (np.random.randint(255, size=(3, 28, 28)), [1,2,3]), ([], [])
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if n is None:
        n = len(x_train)

    for i in range(n):
        yield y_train[i], x_train[i].astype(np.int64), y_train[i]


def main():
    # initializations
    console = Console()  # console for logging
    timer = Timer()  # timer for timing

    # the image image_processor
    cortex = Cortex()
    image_processor = ImageProcessor(cortex=cortex)
    keyboard_processor = KeyboardProcessor(cortex=cortex)

    # get training data
    # images = getSimulateGamePlay()
    images = simulateMNISTDigitRecognition(50)

    # holds the result report
    accuracy = []
    results = {
        'actual': [],
        'predicted': [],
        'probability': [],
        'error': [],
        'accuracy': []
    }

    collected = {x: 0 for x in range(10)}
    cn = 3

    counter = 0  # to hold all the data trained with

    # for data in dataset
    for (image_name, image, metadata) in images:
        if len(results['actual']) == cn*10:
            break

        # skip already satisfied criteria
        if collected[metadata] == cn:
            continue

        counter += 1
        collected[metadata] += 1

        # the dimensions
        if len(image.shape) == 3:
            w, h, d = image.shape

        else:
            (w, h), d = image.shape, 1

        # report image information
        console.log(
            f'LOADING IMAGE_{counter}: {image_name:>7}, initial dimension: width = {w}, height = {h}, depth = {d}, metadata: {metadata}')

        # processor processing results report
        timer.run(lambda: keyboard_processor.run(metadata, verbose=1))
        timer.run(lambda: image_processor.run(image, verbose=1))

        # find the final prediction from agent
        predicted, probability_of_predicted = cortex.result

        # the result of model
        predicted, probability_of_predicted = (metadata, probability_of_predicted) if metadata in predicted else (None, None)

        # re-set the result_vector
        cortex.result = [], 0

        # populate the overal result dictionary
        results['actual'].append(metadata)
        results['predicted'].append(predicted)
        results['probability'].append(probability_of_predicted)
        results['error'].append(int(predicted != metadata))
        results['accuracy'].append(int(predicted == metadata))

        if predicted is not None:
            accuracy.append(int(predicted == metadata))

        console.log('-'*100, '\n')

    # display stats
    console.log('Stats\n============\n')

    # processor stats report
    if image_processor.image_memory_line.data is not None:
        console.log(
            f'{len(image_processor.image_memory_line.data)} image segments discovered and saved into memory')

    # stats report for the key-proc
    if keyboard_processor.keyboard_memory_line.data is not None:
        console.log(
            f'{len(keyboard_processor.keyboard_memory_line.data)} keystroke discovered and saved into memory')

    # cast to a dataframe
    results = pd.DataFrame(results)

    # display final result
    console.log(f'\nFinal Results\n=============\n{results}')

    accuracy1 = sum(accuracy) / len(accuracy) if len(accuracy) > 0 else 1
    accuracy2 = sum(accuracy) / len(results) if len(results) > 0 else 1
    console.log(f'accuracy: local = {accuracy1:.4f}, global = {accuracy2:.4f}')

    # save the sampled training
    np.save('trainX.npy', np.array(cortex.train_images_input_batch))
    np.save('trainY.npy', np.array(cortex.train_images_output_batch))


if __name__ == '__main__':
    main()
