import json
from collections import defaultdict

import pandas as pd

def data_confidence(x):
    return x / (x + 1)

class ProbabilityGraph:
    def __init__(self, idx):
        self.idx = idx
        self.graph = {}
        self.graph_data = {}
        self.graph_score = defaultdict(int)
        self.graph_accuracy = defaultdict(int)
        self.last_update = None

    def update(self, key, expected):
        '''
        update the probabilty count key[total frequency], key-value[co-occurence frequency]
        '''

        if key not in self.graph:
            self.graph[key] = defaultdict(int)
            self.graph_data[key] = ([], [])

        # update the freq counter for the relationship between key-value
        self.graph[key][expected] += 1

    def predict(self, key, expected, context):
        if key not in self.graph:
            return -1, None, -1

        # data collected on context
        data = pd.DataFrame(self.graph[key], index=[0]).T[0]
        score = int(data.idxmax() == expected)

        self.graph_data[key][score].append(context)
        self.graph_score[key] += score
        self.graph_accuracy[key] = (data_confidence(data.sum()) * self.graph_score[key]) / data.sum()

        # normalize for output
        data /= data.sum()
        data *= self.graph_accuracy[key]
        
        # return self.graph_accuracy[key], data.sort_values(ascending=False).to_json(double_precision=4)
        return self.graph_accuracy[key], data.idxmax(), data[data.idxmax()]

    def run(self, key, expected, context):
        # the proposed expectation
        prediction = self.predict(key, expected, context)

        # update pgraph
        self.update(key, expected)

        return prediction

    def write_json(self, filepath, obj):
        with open(filepath, 'w') as f:
            json.dump(obj, f)

    def save(self):
        self.write_json(f'graph_{self.idx}.json', self.graph)
        self.write_json(f'graph_data_{self.idx}.json', self.graph_data)
        self.write_json(f'graph_score_{self.idx}.json', self.graph_score)
        self.write_json(f'graph_accuracy_{self.idx}.json', self.graph_accuracy)
