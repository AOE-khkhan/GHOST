'''
Author: Joshua, Christian r0b0tx
Date: 1 Nov 2018
File: processor.py
Project: GHOST
'''


# import from python standard lib
import re
from itertools import combinations

class Processor:
    """docstring for Processor"""    
    
    def __init__(self):
        self.log_state = True
        self.SIZE = 3

        # holds the dendrites combinations that exist and indexes them
        self.dendrites2index = {}
        self.index2dendrites = {}

        # holds the connection between dendrites
        # and possibly higher order connections (this should represent deep concepts) 
        self.connections2index = [{} for _ in range(self.SIZE)]
        self.index2connections = [{} for _ in range(self.SIZE)]

        # holds the indices of the last dendrites
        self.last_dendrites = None

    def connect(self, d1, d2, level):
        length = len(self.connections2index[level])
        connections = []
        for dx in d1:
            for dy in d2:
                connection = tuple(sorted([dx,dy]))
                print(connection)
                if connection in self.connections2index[level]:
                    connection_index = self.connections2index[level][connection]

                else:
                    self.connections2index[level][connection] = length
                    connection_index = length
                    length += 1

                connections.append(connection_index)
        return connections

    def log(self, output, title=None):
        if not self.log_state:
            return

        if title != None:
            print('\n {} \n{}'.format(title, ''.join(['=' for _ in range(len(title)+2)])))

        print('{}\n'.format(output))
        return

    def getDendrites(self, input_list):
        return [i for i, x in enumerate(input_list) if x == '1']

    def getHighOrderDendrites(self, dendrites):
        '''
        this combines dendrites to see which combination influences next process
        '''
        length = len(dendrites)
        new_dendrites = []
        for r in range(2, length+1):
            combs = combinations(dendrites, r)

            for comb in combs:
                new_dendrites.append(comb)

        new_dendrites.extend(dendrites)
        return new_dendrites

    def process(self, binary_str):
        input_list = list(binary_str)

        # get the dendrites formed
        dendrites = self.getDendrites(input_list)
        # self.log(dendrites, 'dendrites')

        # get high order dendrites
        dendrites = self.getHighOrderDendrites(dendrites)
        
        # logs the dendrites in memory and represent them as indexes
        dendrites = self.setupDendrites(dendrites)

        self.log(dendrites, 'dendrites updated')

        if self.last_dendrites != None:
            connections = self.connect(dendrites, self.last_dendrites, 1)
            self.log(connections, 'connections')


        self.last_dendrites = dendrites
        return


    def setupDendrites(self, dendrites):
        length = len(self.dendrites2index)
        dendrite_indices = []
        for dendrite in dendrites:
            if dendrite not in self.dendrites2index:
                self.dendrites2index[dendrite] = length
                self.index2dendrites[length] = dendrite
                index = length
                length += 1


            else:
                index = self.dendrites2index[dendrite]

            dendrite_indices.append(index)
        return dendrite_indices
