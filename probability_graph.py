import json, time
from collections import defaultdict

import pandas as pd
from modeler import Modeler

def data_confidence(x):
    return x / (x + 1)


class ProbabilityGraph:
    def __init__(self):
        self.idx = f"{time.time():.4f}"

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
            self.graph_models[key] = defaultdict(Modeler)

        # update the freq counter for the relationship between key-value
        self.graph[key][expected] += 1

        # data collected on context
        data = pd.DataFrame(self.graph[key], index=[0]).T[0]
        
        # the total number of references of key
        data_sum = data.sum()

        # if it got it right
        score = int(data.idxmax() == expected)
        
        # id for data storage
        idx = (key, expected)

        # update the score
        self.graph_score[idx] += score

        # accuracy of key
        accuracy = self.graph_score[idx] / data_sum

        # update graph life
        self.graph_life[idx] = (accuracy, int(data_sum))

        # update context and score relationship (for modeler: classifier)
        self.graph_models[key][expected].add_data(context, score)


    def predict(self, key, context):
        if key not in self.graph:
            return None, -1

        # data collected on context
        data = pd.DataFrame(self.graph[key], index=[0]).T[0]

        # data count
        data_sum = data.sum()
        
        # update by the classifier probabilty and use classifier to check if context is a class member
        max_prediction, max_confidence = None, -1

        for prediction, model in self.graph_models[key].items():
            # check for model prediction
            confidence = abs(model.predict(context)) * (data / data_sum) * data_confidence(data_sum)
            
            # check if value can uppdate max
            if confidence < max_confidence:
                continue
            
            # reset max
            max_prediction = prediction
            max_confidence = confidence

        # prediction = (data / data.sum())
        
        return max_prediction, max_confidence


    def write_json(self, filepath, obj):
        with open(filepath, 'w') as f:
            json.dump(obj, f)

    def save(self):
        self.write_json(f'cache/graph_{self.idx}.json', self.graph)
        self.write_json(f'cache/graph_life_{self.idx}.json', self.graph_life)
