import pickle
import numpy as np

from collections import defaultdict
from probability_graph import ProbabilityGraph

class ProbabilityNetwork:
    def __init__(self, context_size, num_of_processes):
        # the size of the context
        self.context_size = context_size

        # the number of sampling to be done
        self.num_of_processes = num_of_processes

        # all pgrapghs
        self.probability_graphs = defaultdict(ProbabilityGraph)
        # [ProbabilityGraph(idx) for idx in range(len(self.context_manager.indices))]

        self.context = ''
        self.context_list = []
        self.context_array = np.array([])
    
    def update_context(self, token):
        # add token to context
        self.context_list.append(token)

        # if size is more than specified then remove oldest
        if len(self.context_list) > self.context_size:
            self.context_list.pop(0)

        # the context string
        self.context = ''.join(self.context_list)
        self.context_array = np.array(self.context_list)

    def run(self, input_value, verbose=0):
        # the previous context information before update
        old_context = self.context
        old_context_array = self.context_array.copy()

        # update the context manager
        self.update_context(input_value)

        # if the context size is up to size (for starters)
        if len(old_context_array) != self.context_size:
            return

        # the initials
        max_confidence, max_prediction = 0, 'VOID'

        # the ones used or called
        processed = set([0])

        # checking if all processing has been achieved
        while len(processed) < self.num_of_processes + 1:
            # indices sampled
            indices = np.where(np.random.randint(2, size=self.context_size) == 1)[0]

            # decimal of indices
            idx = int((2**(self.context_size - indices - 1)).sum())

            if idx in processed:
                continue
            
            # register the sampled indices
            processed.add(idx)

            # select the probability graph
            old_context_token = ''.join(old_context_array[indices])
            self.probability_graphs[idx].update(
                old_context_token, input_value, old_context
            )

            # infer with the pgraph
            context_token = ''.join(self.context_array[indices])
            prediction, confidence = self.probability_graphs[idx].predict(
                context_token, self.context
            )

            # if self.context == 't_7':
            #     print(self.context, context_token, prediction, confidence, self.probability_graphs[idx].graph.get(context_token), sep=' => ')

            if confidence < max_confidence:
                continue
            
            max_confidence = confidence
            max_prediction = prediction

        return f'({max_prediction:>4}, {max_confidence:.4f})'

    def save(self):
        # Store data (serialize)
        with open('cache/probability_graphs.pickle', 'wb') as handle:
            pickle.dump(self.probability_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)

