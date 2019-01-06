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
    
    def __init__(self, state_size=8, size=3):
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

        # holds the last information of size length
        self.context = []

        # concepts weights
        self.transformation_concept_rank = []
        self.transformation_concept_rank_freq = []

        # holds the last concepts
        self.last_concepts = []
        self.last_concept_indices = []

        # holds the hidden relation between states and concepts
        self.node_transformations = []
        self.node_transformation_total_freq = []
        self.node_transformation_freq = []

        # used to model
        self.last_transformations_match = []
        self.last_transformation_concept_used = []

        for _ in range(self.STATE_SIZE):
            self.last_transformation_concept_used.append([])
            self.last_transformations_match.append([])
        
        # for dynamic programming
        self.memoized = {}
        self.MEMOIZED_SIZE = 256

    def addNode(self, concept_model):
        # add concepts
        self.nodes.append(concept_model)

        # matrix of concpts to states
        self.node_state_freq.append([0.0 for _ in range(self.STATE_SIZE)])
        
        # matrix of concepts to transformations        
        self.node_transformations.append([])
        self.node_transformation_freq.append([])
        self.node_transformation_total_freq.append([]) # using this cos transformations are not necessarily mutually exclusive

        self.transformation_concept_rank.append(0)
        self.transformation_concept_rank_freq.append([])

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

    def getConcepts(self, context=None, size=None, reverse=False):
        if context == None:
            context = self.context.copy()
            size = self.SIZE

        if reverse and len(context) < size:
            return []

        val = '0' if reverse else '1'
        concepts = []

        # get concept range based on precent concept as the context might not be up to max
        concept_range = sum([2**x for x in range(len(context))])
        m = size - 1
        for n in range(concept_range):
            bin_li = list(toBin(n+1, size))
            concept = [context[m - i] for i, x in enumerate(bin_li) if x == val]
            if reverse:
                concept.reverse()
            concept = tuple(concept)
            concepts.append(concept)
        return concepts

    def getConceptScoreWeight(self, scores):
        scores_ = scores.copy()
        if type(scores) == dict:
            scores_ = [value for value in scores.values()]
        concept_ranks = self.getRanks(scores_)
        if type(scores) == dict:
            return {score:concept_ranks[scores_.index(scores[score])]/self.CONCEPT_SIZE for score in scores}

        else:
            return [rank/self.CONCEPT_SIZE for rank in concept_ranks]

    def getDataIndices(self, data, dataList):
        return [index+1 for index, value in enumerate(dataList) if value == data]

    def getRanks(self, dataList):
        sortedDataList = sorted(dataList)
        ranks = []
        ranksFound = {}
        for index, value in enumerate(dataList):
            if value in ranksFound:
                rank = ranksFound[value]

            elif dataList.count(value) > 1:
                indices = self.getDataIndices(value, sortedDataList)
                rank = self.mean(indices)

            else:
                rank = sortedDataList.index(value) + 1

            ranksFound[value] = rank
            ranks.append(rank)
        return ranks

    def getMemoized(self, function_name, arguments):
        if function_name not in self.memoized:
            self.memoized[function_name] = {}
            
        if arguments not in self.memoized[function_name]:
            return
        return self.memoized[function_name][arguments]
    
    def getMaxValueIndices(self, processes, return_max_weight=False):
        m = max(processes) if len(processes) > 0 else 0
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
        if len(self.context) < self.SIZE:
            self.addToContext(data)
            return

        # update the node network
        self.update(data)

        # add data to context
        self.addToContext(data)

# -------------------------------------------the process instance-------------------------------------
        mtw = 0

        # the scores for the concept rank
        transformation_concept_rank = {}

        # the transformation variables
        tp, transformation_processes, transformation_processes_rank = [], [], []
        transformations_match, transformation_concept_used = [], []

        #the pdf variables
        pdf_processes = []

        for _ in range(self.STATE_SIZE):
            pdf_processes.append([])

            # holds the transformation values
            tp.append(0) #for the final value of transformation processes
            transformation_processes.append([])
            transformation_processes_rank.append(0) #th transformation concept scores that make up the process

            # the transformation concept used and the models
            transformation_concept_used.append([])
            transformations_match.append([])

        # get the states at this instance
        concepts = self.getConcepts()

        # index of the concepts
        concept_node_indices = []

        for concept_index, concept in enumerate(concepts):
            concept_model = (concept_index, concept)

            if concept_model not in self.nodes:
                self.addNode(concept_model)

            # the id of the concept
            concept_node_index = self.nodes.index(concept_model)

            # save indices
            concept_node_indices.append(concept_node_index)
            
# --------------------------------------------------concept rank--------------------------------------------
            transformation_concept_rank[concept_node_index] = self.transformation_concept_rank[concept_node_index]
                
            # the max states:Purpose is to avoid noise
            concept2states_freq, state_weight = self.getMaxValueIndices(self.normalize(self.node_state_freq[concept_node_index]), True)
            
# ------------------------------------------probability density function---------------------------------------
            for state in concept2states_freq:
                # calculate the total weight of the concept
                weight = state_weight

                # include teh weight in the processes
                pdf_processes[state].append(weight)

                # if len(self.context) > 2 and self.context[-2] == 57 and self.context[-3] == 96 and weight > 0:
                #     print('cm = {}, state = {}, pdf_concept_weight = {} / {} = {} * {} = {}'.format(concept_model, state, node_freq, node_total_freq, pdf_node_weight, state_weight, weight))

# --------------------------------------------------transformations---------------------------------------------------
            transformations = self.node_transformations[concept_node_index]
            tf = self.node_transformation_freq[concept_node_index]      #transformation freq
            ttf = self.node_transformation_total_freq[concept_node_index] #transformation total freq

            # get the transformation_weights
            # transformation_weights = [tf[i] * self.trustFactor(ttf[i]) / ttf[i] if ttf[i] > 0 else 0 for i in range(len(transformations))]
            transformation_weights = [tf[i] / ttf[i] if ttf[i] > 0 else 0 for i in range(len(transformations))]

            # transformation_weights = self.normalize(tf)
            # transformation_weights = [transformation_weights[i] * self.trustFactor(tf) for i in range(len(transformations))]

            max_transformation_weight_ids, max_transformation_weight = self.getMaxValueIndices(transformation_weights, True)

# ==============================================back to transformations===========================
            # for transformation_index in max_transformation_weight_ids:
            for transformation_index in range(len(transformations)):
                

                # the transformation model
                transformation_model = transformations[transformation_index]

                # the tarsnformation model index
                transformation_model_id = (concept_node_index, transformation_index)

                # the ids for the transformations
                concept_x_index, transformation, concept_y_index = transformation_model

                # transform the transformation
                concept_transform = self.solveTransformation(transformation, concepts[concept_x_index])

                concept_transform_model = (concept_y_index, concept_transform)
                if concept_transform_model not in self.nodes:
                    continue

                # concept transf index
                concept_transform_node_index = self.nodes.index(concept_transform_model)

                # the max states:Purpose is to avoid noise
                concept2states_freq, state_weight = self.getMaxValueIndices(self.normalize(self.node_state_freq[concept_transform_node_index]), True)
                
                factor = len(concept2states_freq)**-1
                
                if state_weight == 0:
                    continue

                for state in concept2states_freq:
                    weight = transformation_weights[transformation_index] * state_weight

                    # if self.context[-1] == 53 and self.context[-2] == 96 and (weight >= mtw or state == 54):
                    #     mtw = weight
                    #     print('concept = {}-{}, transf_weight => {} / {} = {} * {} = {}, ct = {}, s= {}-{}'.format(
                    #         concept_index, concepts[concept_index], format(tf[transformation_index], '.3f'), format(ttf[transformation_index], '.3f'), format(transformation_weights[transformation_index], '.3f'), format(state_weight, '.3f'), format(weight, '.3f'), concept_transform_model, transformation_model, state
                    #         )
                    #     )
                    
                    # save the weights
                    transformation_processes[state].append(weight)

                    # to hold the transformation indices of the transformations used
                    transformations_match[state].append(transformation_model_id)

                # increment the total concept-transformation total freq
                ttf[transformation_index] += 1

# =========================influence the pdf and transformations with the concept weights============
        # pdf
        pdf_processes = [max(x) if len(x) > 0 else 0 for x in pdf_processes]
        
        # transformation
        for state, weights in enumerate(transformation_processes):
            value = 0

            for i, weight in enumerate(weights):
                concept_node_index = transformations_match[state][i][0]
                state_rank = transformation_concept_rank[concept_node_index]
                
                if len(weights) > 0 and weight == max(weights):
                    transformation_concept_used[state].append(concept_node_index)

                    if state_rank > transformation_processes_rank[state]:
                        transformation_processes_rank[state] = state_rank

                    value = max(weights)

            tp[state] = value

        # # get the max transformation states
        # max_tp, max_tp_weight = self.getMaxValueIndices(tp ,True)

        # # get the most influencial concept in the max_tp
        # scores = [transformation_processes_score[state] for state in max_tp]
        # max_score = max(scores) if len(scores) > 0 else 0
        
        # transformation_processes = [weight if state in max_tp and transformation_processes_score[state] == max_score else 0 for state, weight in enumerate(tp)]

        if self.context[-1] == 53 and self.context[-2] == 96:
            for concept_node_index in transformation_concept_rank:
                print(concept_node_index, self.nodes[concept_node_index], transformation_concept_rank[concept_node_index])

        transformation_processes = [max(x) if len(x) > 0 else 0 for x in transformation_processes]  
        
        processes = transformation_processes.copy()
        # processes = pdf_processes.copy()
        
        # THE COMBINATION OF PDF AND TRANSFORMATION
        # processes = [self.mean([pdf_processes[i], transformation_processes[i]]) for i in range(self.STATE_SIZE)]

        # get the max vals
        predicted_outputs, max_weight = self.getMaxValueIndices(processes, True)
        
        self.last_concepts = concepts.copy()
        self.last_concept_indices = concept_node_indices.copy()
        self.last_transformations_match = transformations_match.copy()
        self.last_transformation_concept_used = transformation_concept_used.copy()

        if self.context[-1] == 56 and self.context[-2] == 96 and 1 == 12:
            for concept_node_index in transformation_concept_score:
                print(self.nodes[concept_node_index], transformation_concept_score[concept_node_index])

            for state in range(self.STATE_SIZE):
                if state not in [57, 96]:
                    continue
                for concept_node_index in transformation_concept_used[state]:
                    print(state, self.nodes[concept_node_index], transformation_processes_score[state])
        po = []
        return predicted_outputs, max_weight, po

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

    def update(self, data):
        self.resetMemoized('getMaxProbabilityStates')
        max_probability_states = self.getMaxProbabilityStates(data)    

        # the model and concepts used for transformation in prvious process
        last_transformations_match = self.last_transformations_match[data]
        last_transformation_concept_used = self.last_transformation_concept_used[data]

        concept_node_indices_not_used = self.last_concept_indices.copy()

# ======================================increment the transformation================================
        for i, transformation_model_id in enumerate(last_transformations_match):
            concept_node_index, transformation_index = transformation_model_id
            self.node_transformation_freq[concept_node_index][transformation_index] += 1

            if concept_node_index in concept_node_indices_not_used:
                concept_node_indices_not_used.remove(concept_node_index)
        
        # get the max rank of te unusd concepts
        ranks = [self.transformation_concept_rank[concept_node_index] for concept_node_index in concept_node_indices_not_used]
        rank = max(ranks) + 1 if len(ranks) > 0 else 0

# ======================================updating pdf, pdf_concept and creating transformations===============        
        for concept_index, concept in enumerate(self.last_concepts):
            concept_model = (concept_index, concept)
            concept_node_index = self.nodes.index(concept_model)
            
            # the increment for transformation_concept
            transformation_concept_inc = 1 if concept_node_index in last_transformation_concept_used else 0
            
            if concept_node_index not in concept_node_indices_not_used and rank > self.transformation_concept_rank[concept_node_index]:
                self.transformation_concept_rank[concept_node_index] = rank

# =====================================increment the node for pdf===============================
            self.node_state_freq[concept_node_index][data] += 1

# ======================================create transformation weights=========================================
            for cni in self.last_concept_indices:

                for concept_model in max_probability_states:
                    level, other_concept = concept_model
                    factor = len(max_probability_states[concept_model])**-1

                    if factor < 1:
                        continue

                    if not all([x in other_concept for x in concept]):
                        continue

                    transformations = self.getTransformations(concept, other_concept)
                    for transformation in transformations:
                        transformation_model = (concept_index, transformation, level)

                        if transformation_model not in self.node_transformations[cni]:
                            self.node_transformations[cni].append(transformation_model)
                            self.node_transformation_freq[cni].append(0)
                            self.node_transformation_total_freq[cni].append(0)

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
    # td = [learn_counting(11, 3), learn_counting(14, 1)]
    td = [learn_counting(11, 3), learn_counting(21, 2), learn_counting(31, 1)]#, learn_counting(41, 1), learn_counting(51, 1)]#, learn_counting(61, 2), learn_counting(71, 1), learn_counting(101, 1)]
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

            result = PROCESSOR.process(data)
            if result == None:
                outputs, weight, po = None, None, None

            else:
                outputs, weight, po = result
                outputs = [str(chr(x)).encode('utf-8') for x in outputs]
            
                last_outputs = outputs.copy()
            last_input_data = c
if __name__ == '__main__':
    main()

