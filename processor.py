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
from functions import log, formatFloat

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
        self.register = {}  #binary_data:index
        self.registry = {}  #index:binary_data

        # matrix of nodes that process and project data
        self.nodes = { for comb in combinations(for __ in range(self.STATE_SIZE)}

        # holds the last information of size length
        self.context = []

        # contains the output wave function
        self.processes = [[[] for _ in range(self.STATE_SIZE)] for ___ in range(self.SIZE)]

    def addToContext(self, data):
        if len(self.context) == self.SIZE:
            self.context.pop()
        
        self.context.insert(0, data)
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
        binary_data_index = self.save(binary_data)

        # self.log(binary_data_index, 'bin_data_index')

        # reinforce the nodes to make connections in wave forms ( of past data and current one)
        for i, bdi in enumerate(self.context):
            self.nodes[i][bdi][binary_data_index] += 1
        
        # self.log(self.nodes, 'nodes')
        # add the influence of the current data to process
        for i in range(self.SIZE):
            weights = self.normalize(self.nodes[i][binary_data_index])
            # print(i, weights)
            for j in range(self.STATE_SIZE):
                self.processes[i][j].append(weights[j])
        # self.log(self.processes[0], 'processes')

        predicted_outputs = [formatFloat(sum(x)/self.SIZE) for x in self.processes[0]]
        # self.log([formatFloat(sum(x)/self.SIZE) for x in self.processes[0]], 'processes')
        # self.log(predicted_outputs, 'predicted_outputs')
        
        m = formatFloat(max(predicted_outputs))
        predicted_outputs = [self.registry[i] for i, x in enumerate(predicted_outputs) if x >= m and m > formatFloat(0)]
        # self.log(predicted_outputs, 'predicted_outputs')
        # print(self.register, self.registry)
        # predicted_output = predicted_outputs[0] if len(predicted_outputs) > 0 else None

        # self.log(m, 'max')
        # self.log(predicted_output, 'output')

        # restructure the processes
        self.processes.append([[] for _ in range(self.STATE_SIZE)])
        self.processes = self.processes[1:]


        # add data to context
        self.addToContext(binary_data_index)

        return predicted_outputs, m, self.processes

    def save(self, data):
        if data in self.register:
            return self.register[data]

        else:
            length = len(self.register)
            self.register[data] = length
            self.registry[length] = data
            return length