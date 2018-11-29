# -*- coding: utf-8 -*-
"""GHOST.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iMcdaYlNYcGoQJLCharUtCE7Ev2AbfYI
"""

def toBin(c):
    if type(c) == str:
        c = ord(c)
        
    b = bin(c)[2:]
    return ''.join(['0' for _ in range(8 - len(b))]) + b

def toChar(b):
    n = int('0b{}'.format(b), 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode()

def train(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            for s in list(line.strip()):
                yield s
            yield '`'

def learn_counting(n=31, n_iter=2):
    for _ in range(n_iter):
        for i in range(n):
            data = str(i)
            for c in list(data):
                yield c
            yield '`'

def log(output='', title=None):
    if type(output) in [str, int, type(None)]:
        print('{} = {}'.format(title, output))
        return

    if title != None:
        print('\n{} \n{}'.format(title, ''.join(['=' for _ in range(len(title)+2)])))

    print('{}\n'.format(output))
    return


'''
Author: Joshua, Christian r0b0tx
Date: 1 Nov 2018
File: processor.py
Project: GHOST
'''

# import from python standard lib
from itertools import combinations

class Processor:
    """docstring for Processor"""    
    
    def __init__(self, state_size=8, size=3):
        # if to show output
        self.log_state = True

        # processor size in bits
        self.SIZE = size
        self.MEMORY_SIZE = sum([2**x for x in range(self.SIZE)])

        # sensory binary data size
        self.CONCEPT_SIZE = 2**state_size

        # holds the addreses of the sensory state
        self.register = [{} for _ in range(self.MEMORY_SIZE)]  #binary_data:index
        self.registry = [{} for _ in range(self.MEMORY_SIZE)]  #index:binary_data

        # matrix of nodes that process and project data
        self.nodes = [[] for _ in range(self.MEMORY_SIZE)]

        # holds the last information of size length
        self.context = []

        self.last_concepts = []

    def addToContext(self, data):
        if len(self.context) == self.SIZE:
            self.context.pop()
        
        self.context.insert(0, data)
        return

    def getTransformations(self, state_x, state_y):
        states = [state_y]
        for ix, sx in enumerate(state_x):
            new_states = []
            for state in states:
                for iy, sy in enumerate(state):
                    if type(sy) == str:
                        continue
                    
                    if sx == sy:
                        s_new = list(state)
                        s_new[iy] = str(ix)
                        new_states.append(tuple(s_new))
            states = new_states.copy()
        return new_states
        
    def log(self, output, title=None):
        if not self.log_state:
            return

        log(output, title)
        return

    def normalize(self, li):
        s = sum(li)
        if s > 0:
            return [x/s for x in li]

        else:
            return [0 for _ in li]

    def process(self, data):
        for i, concept in enumerate(self.last_concepts):
            index = self.register[i][concept]
            self.nodes[i][index][data] = (1 + self.nodes[i][index][data])/2

            for j, concept_ in enumerate(self.last_concepts):
                if concept != (96,57):
                    continue

                for li, concepts in enumerate(self.nodes):
                    for ci, states in enumerate(concepts):
                        other_concept = self.registry[li][ci]

                        if concept_ != (49,):
                            continue

                        if not all([x in other_concept for x in concept_]):
                            continue

                        max_probability = max(states)
                        if max_probability == 0:
                            continue

                        max_probability_states = [ix for ix, x in enumerate(states) if x == max_probability]
                        if data in max_probability_states:
                            print('concept = {}, level = {}, data = {}'.format(concept, toBin(i+1), data))
                            print('   concept = {}-{}, level = {}, max_vals = {}, max_prob = {}'.format(ci, other_concept, toBin(li+1), max_probability_states, max_probability))
                            self.getTransformations(concept_, other_concept)

        # add data to context
        self.addToContext(data)
        
        processes = [[] for _ in range(self.CONCEPT_SIZE)]

        concepts = []

        # get concept range based on precent concept as the context might not be up to max
        concept_range = sum([2**x for x in range(len(self.context))])
        
        for n in range(concept_range):
            concept = tuple([self.context[i] for i, x in enumerate(list(toBin(n+1))[::-1]) if x == '1'])
            if concept == tuple([]):
                continue

            concepts.append(concept)

        for i, concept in enumerate(concepts):
            if concept in self.register[i]:
                concept_index = self.register[i][concept]

            else:
                length = len(self.register[i])
                self.register[i][concept] = length
                self.registry[i][length] = concept
                concept_index = length
                        
            if concept_index == len(self.nodes[i]):
                self.nodes[i].append([0.0 for _ in range(self.CONCEPT_SIZE)])
            
            # probability distribution of concept to states
            concept2states_weights = self.normalize(self.nodes[i][concept_index])
            
            for state, weight in enumerate(concept2states_weights):
                processes[state].append(weight)
                # if weight > 0:
                #     print(j, weight)

        predicted_outputs = [sum(x)/len(x) if len(x) > 0 else 0 for x in processes]
        m = max(predicted_outputs)
        predicted_outputs = [state for state, x in enumerate(predicted_outputs) if x >= m and m > 0]
        
        self.last_concepts = concepts.copy()                
        return predicted_outputs, m, processes

'''
Author: Joshua, Christian r0b0tx
Date: 1 Nov 2018
File: main.py
Project: GHOST
'''

# initialize objects
PROCESSOR = Processor()

def main():
    # start the sensor
    # input_data = '''hello`hi`i live in nigeria, it is a very big place`hello`hi`what is 1+4?`hello`hi`my name is jack`hi`i am james, i come from the north.`do you speak latin?`no`good morning`hi`how are you doing?`i am great thank you`hello, my name is uri`hi, i'm emma.`hey`hey`can you come today?`no`see me tommorow`hi janet`hello kareem`hello`hi`'''
    # input_data = "hello`hi`two`hello`hi`"
    # input_data = 'hi`hello`1+1 is 2`1+4 is 5`2+3 is 5`5+2 is 7`3+1 is 4`3+4 is 7`2+1 is 3`4+4 is 8`what is 2+3?`5`what is 2+1?`4`what is 3+4?`7`what is 5+2?`7`what is 1+1?`2`hi`'
    
    # input_data = train('train.txt')
    input_data = learn_counting()

    # initialize
    last_outputs = None
    last_input_data = None
    weight = None
    po = None
    
    for c in input_data:
        print('x = {}, y = {}, y_pred = {}, weight = {}, '.format(last_input_data, c, last_outputs, weight, po))

        data = ord(c)

        outputs, weight, po = PROCESSOR.process(data)
        outputs = [chr(x) for x in outputs]
        
        last_outputs = outputs.copy()
        last_input_data = c
if __name__ == '__main__':
    main()

