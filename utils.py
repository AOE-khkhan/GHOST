from random import randint
import numpy as np

def trust_factor(x):
    return x / (x + 1) if x else 0

def transform(token):
    if type(token) == str:
        return ord(token)

    return chr(token)

def simulate_addition(start, stop):
    text = ''
    for num1 in range(start, stop+1):
        for num2 in range(start, stop+1):
            text += f'{num1}+{num2}~{num1+num2}~'
    return text


def simulate_count(start, stop):
    text = f'count_{start}_to_{stop}~'
    for number in range(start, stop+1):
        text += f'{number}~'
    return text

class SimulateGridWorld:
    def __init__(self, shape, agent_token='+', space_token=' '):
        if type(shape) == int:
            shape = (1, shape)

        self.agent_token = agent_token
        self.space_token = space_token

        self.world = np.full(shape, space_token)

        self.current_index = (0, 0)
        self.world[self.current_index] = agent_token


    def __str__(self):
        output = '\n'
        for row in self.world:
            output += ' | '.join(row) + '\n'
            output += '-' * (len(row) * 3) + '\n'
        return output

    def track(self, token):
        if token == 'l':
            step = [0, -1]
        
        elif token == 'r':
            step = [0, 1]

        elif token == 'u':
            step = [-1, 0]

        elif token == 'd':
            step = [1, 0]

        else:
            step = [0, 0]

        return np.array(step)

    def move(self, tracks, callback=lambda self: self):
        for track_token in tracks.lower():
            new_index = np.array(self.current_index) + self.track(track_token)
            
            if not (new_index < 0).any():
                self.world[self.current_index]=self.space_token
                self.current_index=tuple(new_index % np.array(self.world.shape))
                self.world[self.current_index] = self.agent_token

            yield callback(self)
    
if __name__ == "__main__":
    grid_world = SimulateGridWorld(5)

    for world in grid_world.move('r', lambda self: self.world):
        print(world)


