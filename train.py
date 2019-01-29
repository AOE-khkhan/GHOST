import random


def toChar(b):
    n = int('0b{}'.format(b), 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode()

def train(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            for s in list(line.strip()):
                yield s, [s]
            yield ['`'], ['`']

def learn_movement2D(n_iter=10, size=3, start=0, actions=None):
    random_state = True if actions == None else False
    def getAction(i):
        if random_state:
            return actions[random.randint(0, len(actions)-1)]

        else:
            return actions[i]

    # define the position in world
    position = last_position = start

    # get the size of the world
    world = ['_' for __ in range(size)]

    # place the agent in the begining of the world
    world[start] = 'A'
    previous_world = world.copy()

    for i in range(5):
        ret = ['N'] + [x for x in world]
        yield ret, ret

    # get the actions in world
    actions = list('RL') if random_state else actions

    # define conditions in world
    conditions = {'R':[0, 1], 'L':[0, -1]}
    for i in range(n_iter):
        action = getAction(i)
        index, inc = conditions[action]
        pos_temp = position + inc

        if pos_temp in list(range(0, size, 1)):
            # update the position in world
            world[position] = '_'

            position = pos_temp

            world[position] = 'A'

        for action in [action, 'N']:
            ret = [action] + [x for x in previous_world]
            last_position, previous_world = position, world.copy()
            yield ret, ret

def learn_movement3D(n_iter=10, actions=None, size=3, start=[0,0]):
    # define the position in world
    position = start
    last_position = position.copy()

    # get the size of the world
    size = [size, size] if type(size) == int else size
    world = [['_' for __ in range(size[0])] for _ in range(size[0])]

    # place the agent in the begining of the world
    world[start[0]][start[1]] = 'A'

    # get the actions in world
    actions = 'RLUD' if actions == None else actions
    actions = list(actions)

    # define conditions in world
    conditions = {'R':[0, 1], 'L':[0, -1], 'U':[1, 1], 'D':[1, -1]}
    for _ in range(n_iter):
        for __ in actions:
            action = actions[random.randint(0, len(actions)-1)]
            index, inc = conditions[action]
            pos_temp = position[index] + inc

            if pos_temp in list(range(0, size[index], 1)):
                # update the position in world
                world[position[0]][position[1]] = '_'

                position[index] = pos_temp

                world[position[0]][position[1]] = 'A'

            for action in [action, 'N']:
                ret = [action] + [chr(x) for x in last_position]
                last_position = position.copy()
                yield ret, ret#[x for xx in world for x in xx]


def learn_counting(n=101, n_iter=1):
    for i in range(5):
        yield ['`'], ['`']

    for _ in range(n_iter):
        print('\nthis is iteration {} of {} iteration(s): counting to {}\n'.format(_+1, n_iter, n))
        for i in range(n):
            data = str(i)
            for c in list(data):
                yield [c], [c]
            yield ['`'], ['`']

def log(output='', title=None):
    if type(output) in [str, int, type(None)]:
        print('{} = {}'.format(title, output))
        return

    if title != None:
        print('\n{} \n{}'.format(title, ''.join(['=' for _ in range(len(title)+2)])))

    print('{}\n'.format(output))
    return


