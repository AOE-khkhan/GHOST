import numpy as np
from itertools import combinations

class ContextManager:
    def __init__(self, size):
        self.size = size
        size_range = range(size) if type(size) == int else size

        self.indices = [list(combination) for count in size_range for combination in combinations(size_range, count+1)]
        self.contexts = []

        # holds the current n characters
        self.current_token = []
        self.context = ''

    def __str__(self):
        return '\n'.join([f'Context {index}: {context}' for index, context in enumerate(self.contexts)])
    
    def add(self, token):
        self.current_token.append(token)

        if len(self.current_token) < self.size:
            return

        if len(self.current_token) > self.size:
            self.current_token.pop(0)

        self.contexts = []
        current_token_array = np.array(self.current_token)

        for indices in self.indices:
            new_token = ''.join(current_token_array[indices])
            self.contexts.append(new_token)
        
        # the current context in memory
        self.context = ''.join(self.current_token)
