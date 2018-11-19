'''
Author: Joshua, Christian r0b0tx
Date: 1 Nov 2018
File: main.py
Project: GHOST
'''

# import lib code
from processor import Processor

# initialize objects
PROCESSOR = Processor()

def toBin(c):
    b = bin(ord(c))[2:]
    return ''.join(['0' for _ in range(8 - len(b))]) + b

def toChar(b):
    n = int('0b{}'.format(b), 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode()

def train(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            for s in list(line.strip()):
                yield s
            yield '`'

def main():
    # start the sensor
    input_data = '''hello`hi`i live in nigeria, it is a very big place`hello`hi`what is 1+4?`hello`hi`my name is jack`hi`i am james, i come from the north.`do you speak latin?`no`good morning`hi`how are you doing?`i am great thank you`hello, my name is uri`hi, i'm emma.`hey`hey`can you come today?`no`see me tommorow`hi janet`hello kareem`hello`hi`'''

    # input_data = "hello`hi`two`hello`hi`"
    # input_data = "hi`"
    # input_data = 'hi`hello`1+1 is 2`1+4 is 5`2+3 is 5`5+2 is 7`3+1 is 4`3+4 is 7`2+1 is 3`4+4 is 8`what is 2+3?`5`what is 2+1?`4`what is 3+4?`7`what is 5+2?`7`what is 1+1?`2`hi`'
    
    # input_data = train('train.txt')
    last_outputs = None
    last_input_data = None
    weight = None
    po = None
    for data in input_data:
        print('x = {}, y = {}, y_pred = {}, weight = {}, '.format(last_input_data, data, last_outputs, weight, po))

        data_binary_str = toBin(data)

        outputs, weight, po = PROCESSOR.process(data_binary_str)
        outputs = [toChar(x) for x in outputs]
        
        last_outputs = outputs.copy()
        last_input_data = data
if __name__ == '__main__':
    main()