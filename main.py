'''
Author: Joshua, Christian r0b0tx
Date: 1 Nov 2018
File: main.py
Project: GHOST
'''

from train import *
from processor import Processor

# initialize objects for motion
N, S, T = 1, 3, False
# N, S, T = 10, 1, True

# initialize for counting
PROCESSOR = [Processor(n_sensors=N, state_size=8, size=S, no_transformation=T) for _ in range(N)]

def main():
    global N;
    # td = [learn_movement2D(25, size=N-1)]
    td = [learn_counting(15, 3), learn_counting(25, 2), learn_counting(35, 1), learn_counting(55, 1),learn_counting(75, 1)]
    # td = [learn_counting(15, 3), learn_counting(25, 2), learn_counting(35, 1), learn_counting(55, 1),learn_counting(75, 1), learn_counting(105, 1)]
    # td = [learn_counting(21), learn_counting(11), train('train.old.txt'), learn_counting(11), train('train.old.txt')]

    # initialize
    last_outputs, last_input_data, last_weights, last_po, last_sensory_data = [[] for _ in range(N)], [], [None for _ in range(N)], [None for _ in range(N)], [None for _ in range(N)]

    for training_data in td:
        for sensory_data, input_data in training_data:
            weights, po, outputs = [], [], []
            for i, d in enumerate(input_data):
                weights.append([]), po.append([]), outputs.append([]) 

                last_outputs[i] = ['a lot'] if last_weights[i] == 0 or len(last_outputs[i]) > 9 else last_outputs[i]
                print('x = {}-{}, y = {}, y_pred = {}, weight = {}-{},'.format(last_sensory_data[i], last_input_data, input_data, last_outputs[i], last_weights[i], last_po[i]))

                data = ord(d)
                sense_data = list(map(ord, sensory_data))

                result = PROCESSOR[i].process(sense_data, data) 
                if result == None:
                    outputs[i], weights[i], po[i] = [], None, None

                else:
                    outputs[i], weights[i], po[i] = result
                    outputs[i] = [str(chr(x)).encode('utf-8') for x in outputs[i]]
            
            last_weights = weights.copy()
            last_po = po.copy()
            last_outputs = outputs.copy()
            last_sensory_data = sensory_data.copy()
            last_input_data = input_data.copy()
            print()

if __name__ == '__main__':
    main()
