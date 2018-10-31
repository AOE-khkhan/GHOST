import re
from itertools import combinations

def findDataInThread(data, thread=None):
    if thread==None:
        thread == ''

    return re.finditer(data, thread)

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
    
def process(thread):
    last_data = thread.split('.')[-1]
    print('last_data', last_data)
    
    memory_list = thread.split(last_data)
    last_input = memory_list[-2]

    last_input_list = last_input.split('.')[1:-1]
    r = len(last_input_list)
    print('last_input', last_input, last_input_list)

    done = False
    results = []

    parts = getParts(last_input_list)
    for chunk in parts:
        data = '.{}.'.format('.'.join(chunk))

        for i, md in enumerate(memory_list):
            if data in md:
                results.append(i)
                print(md)
##            print('{} => {}'.format(md, data))
            
memory_thread = 'h.i.enter.h.e.l.l.o.enter.1.+.1. .i.s. .2.enter.1.+.4. .i.s. .5.enter.2.+.3. .i.s. .5.enter.5.+.2. .i.s. .7.enter.3.+.1. .i.s. .4.enter.3.+.4. .i.s. .7.enter.2.+.1. .i.s. .3.enter.4.+.4. .i.s. .8.enter.w.h.a.t. .i.s. .2.+.3.?.enter.5.enter.w.h.a.t. .i.s. .2.+.1.?.enter.4.enter.w.h.a.t. .i.s. .3.+.4.?.enter.7.enter.w.h.a.t. .i.s. .5.+.2.?.enter.7.enter.w.h.a.t. .i.s. .1.+.1.?.enter.2.enter.w.h.a.t. .i.s. .1.+.4.?.enter'
process(memory_thread)
