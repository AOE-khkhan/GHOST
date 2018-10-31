import re
from itertools import combinations

memory_thread = 'hi`hello`1+1 is 2`1+4 is 5`2+3 is 5`5+2 is 7`3+1 is 4`3+4 is 7`2+1 is 3`4+4 is 8`what is 2+3?`5`what is 2+1?`4`what is 3+4?`7`what is 5+2?`7`what is 1+1?`2`'
memory_weight = [0 for _ in memory_thread]

def findDataInThread(data, thread=None):
    if thread==None:
        thread == ''

    return re.finditer(re.escape(data), thread)

def getPositions(data, thread=None):
    matches = findDataInThread(data, memory_thread)
    for match in matches:
        start, stop = match.span()
        yield start, stop

def getParts(li):
    length = len(li)
    
    for r in range(length, 0, -1):
        n = length - r + 1
        for a in range(n):
            part = li[a:a+r]
            yield part

def getScore(po):
    pos = []
    for index, p in enumerate(po):
        pos.append({})
        for output in p:
            pos[-1][output] = sum([memory_weight[i] for i in p[output]])
    return pos

def process(data, possible_outputs):
    global models_used, N, last_data
    if last_data != None and models_used != None:
        for output in models_used:
            inc = -1
            if output == data:
                inc = 1

            models = models_used[output]
            for model_id in models:
                memory_weight[model_id] += inc

    positions = getPositions(data, memory_thread)
    index = memory_length = len(memory_thread)

    possible_outputs = possible_outputs[25-N:]
    for i, j in positions:
        if j >= memory_length:
            continue
        possible_output = memory_thread[j:j+N]
        # print('input = {}, matches = {}, output = {}'.format(data, memory_thread[i-1:j], possible_output))

        for n, po in enumerate(possible_output):
            if po not in possible_outputs[n]:
                possible_outputs[n][po] = []

            possible_outputs[n][po].append(i)
    
    print(data)
    possible_outputs_scores = getScore(possible_outputs.copy())
    for p in possible_outputs_scores:
        print(p)

    models_used = possible_outputs[0]

input_data = "hi`hi`what is 1+4?`hello1`hi`my name is jack`hi`"
N = 25
possible_outputs = [{} for _ in range(N)]
last_data = None
models_used = None
for data in input_data:
    # print(possible_outputs)
    process(data, possible_outputs)
    N -= 1
    last_data = data


if __name__ == '__main__':
    main()