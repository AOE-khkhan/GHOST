'''
Author: Joshua, Christian r0b0tx
Date: 1 Nov 2018
File: processor.py
Project: GHOST
'''


# import from python standard lib
import re

class Processor:
    """docstring for Processor"""    
    SIZE = 16
    
    def __init__(self):
        self.memory = ''
        self.memory_length = len(self.memory)
        self.weights = {}
        
        self.last_data = None    #holds the last input
        self.last_reinforced_weights = None  #holds a list of data that reinforced last input

        self.possible_outputs = ({} for _ in range(self.SIZE))

    def findDataInMemory(self, data, memory=None):
        if memory == None:
            memory = self.memory

        return re.finditer(re.escape(data), memory)

    def getPositions(self, data, memory=None):
        matches = self.findDataInMemory(data, memory)
        for match in matches:
            start, stop = match.span()
            yield start, stop

    def getScore(self, po, current_data_id):
        '''
        po - possible outputs
        pos - possible outputs scores
        '''
        pos = []
        for index, p in enumerate(po):
            pos.append({})
            for output in p:
                pos[-1][output] = sum([self.getWeight((matched_data_id, id_n)) for matched_data_id, id_n in p[output]])
        return pos

    def getWeight(self, weight_id):
        if weight_id in self.weights:
            return self.weights[weight_id]

        else:
            self.weights[weight_id] = 0
            return 0

    def mean(self, collections):
        li = collections.copy()
        length = len(li)
        if length == 0:
            return 0

        else:
            if type(li) == dict:
                return sum([x for x in li.values()])/length

            return sum(li)/length

    def process(self, current_data):
        # save data
        current_data_id = self.saveData(current_data)

        # reinforce weights
        if self.last_data != None and self.last_reinforced_weights != None:
            for output in self.last_reinforced_weights:
                inc = 0
                if output == current_data:
                    inc = 1

                weight_ids = self.last_reinforced_weights[output]
                for weight_id in weight_ids:
                    # print(self.getWeight(weight_id), inc, current_data, output, weight_id)
                    self.updateWeight(weight_id, inc)

        positions = self.getPositions(current_data)
        index = memory_length = len(self.memory)

        # gather connections
        possible_outputs = list(self.possible_outputs)[1:] + [{}]
        for i, j in positions:
            if j >= memory_length:
                continue

            possible_output = self.memory[j:j + self.SIZE]
            # print('input = {}, matches = {}, output = {}'.format(current_data, memory_thread[i-1:j], possible_output))

            for n, po in enumerate(possible_output):
                if po not in possible_outputs[n]:
                    possible_outputs[n][po] = []

                possible_outputs[n][po].append((i, n))
        
        possible_outputs_scores = self.getScore(possible_outputs, current_data_id)
        # for p in possible_outputs_scores:
        #     print(p)
        #     break

        # set next state parameters
        self.last_reinforced_weights = possible_outputs[0]
        self.possible_outputs = possible_outputs.copy()
        self.last_data = current_data

        predicted_outputs = possible_outputs_scores[0]
        return {output:predicted_outputs[output] for output in predicted_outputs if predicted_outputs[output] > self.mean(predicted_outputs)}

    def saveData(self, data):
        self.memory += '{}'.format(data)
        self.memory_length += 1
        return self.memory_length

    def updateWeight(self, weight_id, inc):
        mw = self.getWeight(weight_id)
        self.weights[weight_id] += inc
        return