import json
from collections import defaultdict

import pandas as pd
from modeler import Modeler

def data_confidence(x):
    return x / (x + 1)

class ProbabilityGraph:
    def __init__(self, idx):
        # the identity of the graph
        self.idx = idx

        # the graph itself
        self.graph = {}

        # the count of correct score
        self.graph_score = defaultdict(int)

        # the life of the graph
        self.graph_life = {}

        # the models that fit probability for graph input
        self.graph_models = {}

    def update(self, key, expected, context):
        '''
        update the probabilty count key[total frequency], key-value[co-occurence frequency]
        '''

        if key not in self.graph:
            self.graph[key] = defaultdict(int)
            self.graph_models[key] = Modeler(batch_size=2)

        # update the freq counter for the relationship between key-value
        self.graph[key][expected] += 1

        # data collected on context
        data = pd.DataFrame(self.graph[key], index=[0]).T[0]
        
        # the total number of references of key
        data_sum = data.sum()

        # if it got it right
        score = int(data.idxmax() == expected)

        # update the score
        self.graph_score[key] += score

        # accuracy of key
        accuracy = self.graph_score[key] / data_sum

        # update graph life
        self.graph_life[key] = (accuracy, int(data_sum))

        # update context and score relationship (for modeler: classifier)
        self.graph_models[key].add_data(context, score)


    def predict(self, key, context):
        if key not in self.graph:
            return None, -1

        # data collected on context
        data = pd.DataFrame(self.graph[key], index=[0]).T[0]

        # update by the classifier probabilty and use classifier to check if context is a class member
        prediction = abs(self.graph_models[key].predict(context)) * (data / data.sum())
        
        return prediction.idxmax(), prediction[prediction.idxmax()]


    def write_json(self, filepath, obj):
        with open(filepath, 'w') as f:
            json.dump(obj, f)

    def save(self):
        self.write_json(f'graph_{self.idx}.json', self.graph)
        self.write_json(f'graph_life_{self.idx}.json', self.graph_life)
