

# import from python standard lib
import re

class Processor:
    """docstring for Processor"""    
    SIZE = 16
    
    def __init__(self):
        self.memory = 'hi`hello`1+1 is 2`1+4 is 5`2+3 is 5`5+2 is 7`3+1 is 4`3+4 is 7`2+1 is 3`4+4 is 8`what is 2+3?`5`what is 2+1?`4`what is 3+4?`7`what is 5+2?`7`what is 1+1?`2`'
        self.memory_length = len(self.memory)
        self.weights = {}
        
        self.last_data = None    #holds the last input
        self.last_reinforced_weights = None  #holds a list of data that reinforced last input

        self.possible_outputs = [{} for _ in range(self.SIZE)]

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

    def process(self, current_data):
        self.counter = (self.counter+1)
        # save data
        current_data_id = self.saveData(current_data)

        # reinforce weights
        if self.last_data != None and self.last_reinforced_weights != None:
            for output in self.last_reinforced_weights:
                inc = -1
                if output == current_data:
                    inc = 1

                weight_ids = self.last_reinforced_weights[output]
                for weight_id in weight_ids:
                    print(self.getWeight(weight_id), inc, current_data, output, weight_id)
                    self.updateWeight(weight_id, inc)

        positions = self.getPositions(current_data)
        index = memory_length = len(self.memory)

        # gather connections
        possible_outputs = self.possible_outputs[1:] + [{}]
        for i, j in positions:
            if j >= memory_length:
                continue
            
            possible_output = self.memory[j:j + self.SIZE]
            # print('input = {}, matches = {}, output = {}'.format(current_data, memory_thread[i-1:j], possible_output))

            for n, po in enumerate(possible_output):
                if po not in possible_outputs[n]:
                    possible_outputs[n][po] = []

                possible_outputs[n][po].append((i, n))
        
        print(current_data)
        possible_outputs_scores = self.getScore(possible_outputs, current_data_id)
        # for p in possible_outputs_scores:
        #     print(p)
        #     break

        # set next state parameters
        self.last_reinforced_weights = possible_outputs[0]
        self.possible_outputs = possible_outputs.copy()
        self.last_data = current_data

        return possible_outputs_scores[0]

    def saveData(self, data):
        self.memory += ''.format(data)
        self.memory_length += 1
        return self.memory_length

    def updateWeight(self, weight_id, inc):
        mw = self.getWeight(weight_id)
        self.weights[weight_id] += inc
        return