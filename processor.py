'''
Author: Joshua, Christian r0b0tx
Date: 1 Nov 2018
File: processor.py
Project: GHOST
'''


# import from python standard lib
import re
from itertools import combinations

# import from my framework
from functions import log

class Processor:
    """docstring for Processor"""    
    
    def __init__(self, n_sensors=8, size=3):
        # if to show output
        self.log_state = True

        # processor size in bits
        self.SIZE = size

        # sensory binary data size
        self.STATE_SIZE = 2**n_sensors

        # holds the addreses of the sensory state
        self.register = [{} for _ in range(self.SIZE)]  #binary_data:index
        self.registry = [{} for _ in range(self.SIZE)]  #index:binary_data

        # matrix of nodes that process and project data
        self.nodes = [[[0 for _ in range(self.STATE_SIZE)] for __ in range(self.STATE_SIZE)] for ___ in range(self.SIZE)]

        # holds the last information of size length
        self.context = [[] for _ in range(self.SIZE)]

        # contains the output wave function
        self.processes = [[[0 for _ in range(self.STATE_SIZE)] for __ in range(self.STATE_SIZE)] for ___ in range(self.SIZE)]

    def addToContext(self, data, level):
        if len(self.context[level]) == self.SIZE:
            self.context[level].pop()
        
        self.context[level].insert(0, bdi)
        return

    def log(self, output, title=None):
        if not self.log_state:
            return

        log(output, title)
        return

    def normalize(self, li, factor=None):
        s = sum(li)

        if s == 0:
            return [0 for _ in li]

        else:
            return [x/s for x in li]

    def process(self, binary_data):
        # self.log(binary_data, 'bin_data')

        # save the sensoy state and get the index
        binary_data_index = self.save(binary_data, 0)

        # self.log(binary_data_index, 'bin_data_index')

        # add data to context
        self.addToContext(binary_data_index)

        # reinforce the nodes to make connections in wave forms ( of past data and current one)
        for i, bdi in enumerate(self.context):
            self.nodes[i][bdi][binary_data_index] += 1
        
        # self.log(self.nodes, 'nodes')
        # add the influence of the current data to process
        for i in range(self.SIZE):
            weights = self.normalize(self.nodes[i][binary_data_index])
            for j in range(self.STATE_SIZE):
                self.processes[i][binary_data_index][j] += weights[j]

        predicted_outputs = [x/self.SIZE for x in self.processes[0][binary_data_index]]
        # self.log(self.processes[0][binary_data_index], 'processes')
        # self.log(predicted_outputs, 'predicted_outputs')
        
        m = max(predicted_outputs)
        predicted_outputs = [self.registry[i] for i, x in enumerate(predicted_outputs) if m == x and m > 0]
        # predicted_output = predicted_outputs[0] if len(predicted_outputs) > 0 else None

        # self.log(m, 'max')
        # self.log(predicted_output, 'output')

        # restructure the processes
        self.processes.append([[0 for _ in range(self.STATE_SIZE)] for __ in range(self.STATE_SIZE)])
        self.processes = self.processes[1:]

        return predicted_outputs, m

    def save(self, data, level):
        if data in self.register[level]:
            return self.register[level][data]

        else:
            length = len(self.register[level])
            self.register[level][data] = length
            self.registry[level][length] = data
            return length