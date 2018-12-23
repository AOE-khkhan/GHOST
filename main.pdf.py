# -*- coding: utf-8 -*-
"""GHOST.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iMcdaYlNYcGoQJLCharUtCE7Ev2AbfYI
"""
# from numba import jit

def toBin(c, n=8):
    if type(c) == str:
        c = ord(c)
        
    b = bin(c)[2:]
    return ''.join(['0' for _ in range(n - len(b))]) + b

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

def learn_counting(n=101, n_iter=1):
    for _ in range(n_iter):
        print('\nthis is iteration {} of {} iteration(s): counting to {}\n'.format(_+1, n_iter, n))
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
    
    def __init__(self, state_size=8, size=4):
        # if to show output
        self.log_state = True

        # processor size in bits
        self.SIZE = size
        self.CONCEPT_SIZE = sum([2**x for x in range(self.SIZE)])

        # the number of methods used: pdf and transformations
        self.N_METHODS = 2

        # sensory binary data size
        self.STATE_SIZE = 2**state_size

        # matrix of nodes that process and project data
        self.nodes = []
        self.node_state_freq = []
        self.node_state_total_freq = [] #containing: node, transformation_weight, pdf_infl, transformation_infl

        # holds the last information of size length
        self.context = []

        # holds the last concepts
        self.last_concepts = []

        # holds the hidden relation between states and concepts
        self.transformations = []
        self.transformation_nodes = []
        self.transformation_freq = []
        self.transformation_total_freq = []

        self.transformation_other = []
        self.transformation_other_nodes = []
        self.transformation_other_freq = []
        self.transformation_other_total_freq = []

        # used to model
        self.pdf_models_used = [[] for _ in range(self.STATE_SIZE)]
        self.transformation_models_used = [[] for _ in range(self.STATE_SIZE)]

        # for dynamic programming
        self.memoized = {}
        self.MEMOIZED_SIZE = 256

        # mean amplitude overtime
        self.pdf_mean_amplitude = 0

    def addToContext(self, data):
        if len(self.context) == self.SIZE:
            self.context.pop()
        
        self.context.insert(0, data)
        return

    def getMaxProbabilityStates(self, data):
        ret = self.getMemoized('getMaxProbabilityStates', (data))
        if ret != None:
            return ret

        mps = {}
        for concept_index, concept in enumerate(self.nodes):
            states = self.node_state_freq[concept_index]
            max_probability = max(states)
            max_probability_states = [ix for ix, x in enumerate(states) if x == max_probability]
            if data in max_probability_states:
                mps[concept] = max_probability_states

        self.updateMemoized('getMaxProbabilityStates', (data), mps)
        return mps

    def getConcepts(self, reverse=False):
        if reverse and len(self.context) < self.SIZE:
            return []

        val = '0' if reverse else '1'
        concepts = []

        # get concept range based on precent concept as the context might not be up to max
        concept_range = sum([2**x for x in range(len(self.context))])
        m = self.SIZE - 1
        for n in range(concept_range):
            bin_li = list(toBin(n+1, self.SIZE))
            concept = [self.context[m - i] for i, x in enumerate(bin_li) if x == val]
            if reverse:
                concept.reverse()
            concept = tuple(concept)
            concepts.append(concept)
        return concepts

    def getMemoized(self, function_name, arguments):
        if function_name not in self.memoized:
            self.memoized[function_name] = {}
            
        if arguments not in self.memoized[function_name]:
            return
        return self.memoized[function_name][arguments]
    
    def getPredictedOutputs(self, processes, return_max_weight=False):
        m = max(processes)
        predicted_outputs = [state for state, x in enumerate(processes) if x == m]
        if return_max_weight:
            return predicted_outputs, m

        else:
            return predicted_outputs

    def getTransformations(self, concept_x, concept_y):
        concepts = [concept_y]
        for ix, cx in enumerate(concept_x):
            new_concepts = []
            for concept in concepts:
                for iy, cy in enumerate(concept):
                    if type(cy) == str:
                        continue
                    
                    if cx == cy:
                        c_new = list(concept)
                        c_new[iy] = str(ix)
                        new_concepts.append(tuple(c_new))
            concepts = new_concepts.copy()
        return new_concepts
        
    def log(self, output, title=None):
        if not self.log_state:
            return

        log(output, title)
        return

    def mean(self, li):
        if len(li) > 0:
            return sum(li)/len(li)

        else:
            return 0

    def normalize(self, li):
        s = sum(li) if type(li) != dict else sum(li.values())
        if s > 0:
            if type(li) == dict:
                return {x:li[x]/s for x in li}

            else:
                return [x/s for x in li]

        else:
            if type(li) == dict:
                return {x:0 for x in li}

            else:
                return [0 for _ in li]

    # @jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
    def process(self, data):
        # update the node network
        self.update(data)

        # add data to context
        self.addToContext(data)

# ----------------------the process instance-------------------------------------
        # the processes construct
        # the prob densities and the inluences
        pdf_processes = [[] for _ in range(self.STATE_SIZE)]
        p = [[] for _ in range(self.STATE_SIZE)]
        
        # the transf infl and densities
        transformation_processes = [[] for _ in range(self.STATE_SIZE)]
        
        # used to model
        self.pdf_models_used = [[] for _ in range(self.STATE_SIZE)]
        self.transformation_models_used = [[] for _ in range(self.STATE_SIZE)]

        # get the states at this instance
        concepts = self.getConcepts()
        reverse_concepts = self.getConcepts(True)

        for concept_index, concept in enumerate(concepts):
            concept_model = (concept_index, concept)
            if concept_model not in self.nodes:
                self.nodes.append(concept_model)
                self.node_state_freq.append([0.0 for _ in range(self.STATE_SIZE)])
                self.node_state_total_freq.append([0.0 for _ in range(self.STATE_SIZE)])

            # the id of the concept
            concept_node_index = self.nodes.index(concept_model)

            concept2states_freq = self.node_state_freq[concept_node_index]            
# ---------------------------------------------------------------probability density function--------------
            for state, state_freq in enumerate(concept2states_freq):
                # pdf weight for relative value
                node_state_total_freq = self.node_state_total_freq[concept_node_index][state]
            
                state_weight = state_freq/node_state_total_freq if node_state_total_freq > 0 else 0
                pdf_processes[state].append(state_weight * self.trustFactor(state_freq) * len(concept) / self.SIZE)
                p[state].append(state_weight * self.trustFactor(state_freq))

                # increment the node
                self.node_state_total_freq[concept_node_index][state] += 1
                self.pdf_models_used[state].append(concept_node_index)
# --------------------------------------------------transformations---------------------------------------------------
            
        
        # processes = [max(x) for x in pdf_processes]
        # processes = self.normalize(pdf_processes_sum)
        # processes = [(self.mean(x)+pdf_processes_norm[i])/2 for i, x in enumerate(pdf_processes)]

        # processes = [self.mean(x) for x in pdf_processes]
        # transformation_processes = [self.mean([xx for xx in x if xx >= self.mean(x)]) for x in transformation_processes]
        processes = [sum(x) for x in pdf_processes]
        processes = self.normalize(processes)

        po = [self.mean([xx for xx in x if xx >= self.mean(x)]) for x in p]
        # po = self.normalize(po)
        po = [(format(x, '.2f'), chr(i)) for i, x in enumerate(po) if x > 0]
        # self.pdf_models_used = [[self.pdf_models_used[state][i] for i, xx in enumerate(x) if xx >= processes[state]] for state, x in enumerate(pdf_processes)]
        # processes = self.normalize(processes)
        # processes = [self.mean([pdf_processes[i], transformation_processes[i]]) for i in range(self.STATE_SIZE)]

        predicted_outputs, max_weight = self.getPredictedOutputs(processes, True)
        
        self.last_concepts = concepts.copy()
        # return predicted_outputs, format(max_weight, '0.2f'), [self.mean([xx for xx in p[x] if xx >= self.mean(p[x])]) for i, x in enumerate(predicted_outputs)]
        return predicted_outputs, format(max_weight, '0.2f'), po

    def resetMemoized(self, function_name):
        if function_name not in self.memoized:
            self.memoized[function_name] = {}
            return

        self.memoized.pop(function_name)
        self.memoized[function_name] = {}
        return 
            
    def solveTransformation(self, transformation, concept):
        transform = list(transformation)
        for i, t in enumerate(transformation):
            if type(t) != str:
                continue

            index = int(t)
            transform[i] = concept[index]
        return tuple(transform)

    def trustFactor(self, x):
        if type(x) == list:
            x = len(x)
        return x / (1 + x) 
        # return (1 + ()) 

    # @jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
    def update(self, data):
        for concept_node_index in self.pdf_models_used[data]:
            self.node_state_freq[concept_node_index][data] += 1
        
        return

    def updateMemoized(self, function_name, arguments, value):
        if function_name not in self.memoized:
            self.memoized[function_name] = {}

        vals = self.memoized[function_name]
        length = len(vals)
        if length == self.MEMOIZED_SIZE:
            self.memoized[function_name].pop([x for x in vals.keys()][random.randint(length)])

        self.memoized[function_name][arguments] = value
        return

'''
Author: Joshua, Christian r0b0tx
Date: 1 Nov 2018
File: main.py
Project: GHOST
'''

# initialize objects
PROCESSOR = Processor()

def main():
    # td = [learn_counting(101)]
    td = [learn_counting(11, 10), learn_counting(21, 3), learn_counting(31, 2), learn_counting(51, 1), learn_counting(53, 1)]
    # td = [learn_counting(11, 3), learn_counting(21, 3), learn_counting(31, 2), learn_counting(51, 2), learn_counting(61), learn_counting(101)]
    # td = [learn_counting(21), learn_counting(11), train('train.old.txt'), learn_counting(11), train('train.old.txt')]

    # initialize
    last_outputs = []
    last_input_data = None
    weight = 0
    po = None
    
    for training_data in td:
        for c in training_data:
            if weight == 0 or len(last_outputs) > 9:
                last_outputs = ['a lot']
            print('x = {}, y = {}, y_pred = {}, weight = {}-{}, '.format(last_input_data, c, last_outputs, weight, po))

            data = ord(c)

            outputs, weight, po = PROCESSOR.process(data)
            outputs = [str(chr(x)).encode('utf-8') for x in outputs]
            
            last_outputs = outputs.copy()
            last_input_data = c
if __name__ == '__main__':
    main()

