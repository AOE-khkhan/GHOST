import pickle
import numpy as np
import pandas as pd

from collections import defaultdict, Counter
from utils import trust_factor

class ProbabilityModel:
    def __init__(self, context_size, models_garbage_batch):
        # the muliple of memory size the model should collect garbage
        self.MODELS_GARBAGE_BATCH = models_garbage_batch

        # the size of the context
        self.context_size = context_size

        # the contexts
        self.context_list = []
        self.context_array = np.array([])

        # the data points
        self.x = []
        self.y = []
        
        # the models (patterns)
        self.models = {}
        self.models_index = [defaultdict(set) for _ in range(self.context_size)]
        self.models_distribution = {}

        # the last predictions (models)
        self.last_prediction = []

    def create_model(self, best_context_indices, distribution, expectation):
        # create and save the model
        indices = tuple(best_context_indices)
        tokens = tuple(self.context_array[best_context_indices])

        # the model key
        model_key = (indices, tokens)

        # create and save model if not existing
        if model_key not in self.models:
            self.models[model_key] = defaultdict(int)
            self.models_distribution[model_key] = distribution

            for index, token in zip(indices, tokens):
                self.models_index[index][token].add(model_key)
 
        # update the expectation for model
        self.models[model_key][expectation] += 1

    def discover_pattern(self, expectation, context_indices=None, context_tokens=None):
        # the expectations calculate
        confidences = []

        for context_index, context_token in enumerate(self.context_array):
            # get the probability with respect to the expectation
            expectation_probability = self.get_conditional_probability(
                expectation, [context_index], [context_token]
            )

            # save to expectation list
            confidences.append(expectation_probability)

        # all confidence and indices in descending order
        confidences = np.array(confidences)
        order = confidences.argsort()[::-1]

        # highest assurance
        max_confidence = max(confidences)
        best_context_indices = [order[0]]
        best_context_tokens = [self.context_array[order[0]]]

        # check custom contexts
        for index in range(2, self.context_size+1):
            # the custom context
            context_indices = order[:index]
            context_tokens = self.context_array[context_indices]

            # calculate custom context confidence
            custom_context_confidence = self.get_conditional_probability(
                expectation, context_indices, context_tokens
            )

            if custom_context_confidence < max_confidence:
                break

            else:
                best_context_indices = context_indices
                best_context_tokens = context_tokens
                max_confidence = custom_context_confidence

        # make indices into array and sort
        best_context_indices = np.array(best_context_indices)
        best_context_indices.sort()

        # obtain data subset
        x0 = self.input_array[(self.input_array[:, best_context_indices] == best_context_tokens).all(1)]
        
        # token distribution for uncertain bits
        distribution = []

        # create distribution of uncertain occurences
        for i in range(self.context_size):
            # count the number of occurences
            c0 = Counter(x0[:, i])
            c0 = {key: [value, value] for key, value in c0.items()}

            # the distr of occurences in model
            distribution.append(c0)

        # return the indices as a numpy array for easy indexing of important context token
        return best_context_indices, tuple(distribution), max_confidence

    def get_conditional_probability(self, expectation, context_indices, context_tokens):

        # obtain data subset
        y0 = self.output_array[(
            self.input_array[:, context_indices] == context_tokens).all(1)]

        # calculate the occurance and relative (trsufactor) probability
        c0 = pd.DataFrame(Counter(y0), index=[0]).T[0]
        p0 = (trust_factor(c0.sum()) * c0) / c0.sum()

        return p0.get(expectation, 0)

    def predict(self):
        # clear out last predictions
        self.last_prediction = []

        ''' predict the expectation using the current models '''
        max_prediction, max_confidence = None, 0

        # all models that are close to the context
        processed_model_keys = set()

        # get all models that relate to tokens in context
        for index, token in enumerate(self.context_list):
            model_keys = self.models_index[index][token]

            # check for the model that can infer from context
            for indices, tokens in model_keys:
                # the identity of model
                model_key = (indices, tokens)

                if model_key in processed_model_keys:
                    continue

                processed_model_keys.add(model_key)

                if not (self.context_array[np.array(indices)] == np.array(tokens)).all():
                    continue

                # the expectations
                expectation = self.models.get(model_key, None)

                if expectation is None:
                    continue
                
                # analyse the expectations
                expectation = pd.DataFrame(expectation, index=[0]).T[0]
                expectation_size = expectation.sum()

                # normalize the expectation
                expectation = (trust_factor(expectation_size) * expectation) / expectation_size

                # remove model key when the accuarcy is chaotic
                if expectation_size and not expectation_size % self.MODELS_GARBAGE_BATCH and not (expectation > 0.5).any():
                    model_keys.remove(model_key)
                    self.models.pop(model_key, None)
                    self.models_distribution.pop(model_key, None)
                    continue

                # the distribution of toekns for model
                distribution = self.models_distribution[model_key]

                distr = [] #temp list of all attribution ratio to model based on token
                for index, token in enumerate(self.context_list):
                    n, N = distribution[index].get(token, [0, 0])
                    attribution = (trust_factor(N) * n) / N if N else 0

                    distr.append(attribution)

                # probability of context being in the model class
                in_class = sum(distr) / len(distr) if len(distr) else 0

                # infer prediction
                prediction = expectation.idxmax()
                confidence = in_class * expectation[prediction]

                # save this inference to update in next inference
                self.last_prediction.append((model_key, prediction))

                # pick the bext prediction
                if confidence < max_confidence:
                    continue

                max_confidence = confidence
                max_prediction = prediction

        return max_prediction, max_confidence

    def run(self, input_value):
        # update the last models based on the new info
        self.update_model(input_value)

        # the initial best context token found
        best_context_indices, confidence = np.array([]), 0

        # if the context is partial
        if len(self.context_list) < self.context_size:
            self.update_context(input_value)  # update the context manager
            return best_context_indices, confidence #do nothing except return deafult

        # update data
        self.x.append(self.context_list.copy())
        self.y.append(input_value)

        # make data into array
        self.input_array  = np.array(self.x, dtype='int')
        self.output_array = np.array(self.y, dtype='int')

        # update the network
        model_context_indices, model_distribution, model_confidence = self.discover_pattern(input_value)
        
        # return best context token found
        if model_confidence > 0.5:
            # update the models
            self.create_model(model_context_indices, model_distribution, input_value)

        # update the context manager
        self.update_context(input_value)
        
        # try to predict using curent models discovered
        prediction, confidence = self.predict()

        return prediction, confidence

    def save(self):
        # Store data (serialize)
        with open('cache/probability_model.pickle', 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def update_context(self, token):
            # add token to context
        self.context_list.append(token)

        # if size is more than specified then remove oldest
        if len(self.context_list) > self.context_size:
            self.context_list.pop(0)

        # the context string
        self.context_array = np.array(self.context_list)

    def update_model(self, input_value):
        for model_key, prediction in self.last_prediction:
            score = int(input_value == prediction)

            for index, token in enumerate(self.context_list):
                if token not in self.models_distribution[model_key][index]:
                    self.models_distribution[model_key][index][token] = [0, 0]

                # update probability count
                self.models_distribution[model_key][index][token][0] += score
                self.models_distribution[model_key][index][token][1] += 1

            self.models[model_key][input_value] += 1
