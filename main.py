'''
Author: Joshua, Christian r0b0tx
Date: 1 Nov 2018
File: main.py
Project: GHOST
'''

# import lib code
from processor import Processor
from functions import log

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
            line = line.strip()

            if line == '':
                continue

            for s in list(line):
                yield s
            yield '`'

def main():
    # start the sensor
    input_data = '''hello`hi`i live in nigeria, it is a very big place`hello`hi`what is 1+4?`hello`hi`my name is jack`hi`i am james, i come from the north.`do you speak latin?`no`good morning`hi`how are you doing?`i am great thank you`hello, my name is uri`hi, i'm emma.`hey`hey`can you come today?`no`see me tommorow`hi janet`hello kareem`hello`'''

    # input_data = "ho`hi`ho`hi`"
    # input_data = "hi`hello`two`hi`"
    # input_data = 'hi`hello`1+1 is 2`1+4 is 5`2+3 is 5`5+2 is 7`3+1 is 4`3+4 is 7`2+1 is 3`4+4 is 8`what is 2+3?`5`what is 2+1?`4`what is 3+4?`7`what is 5+2?`7`what is 1+1?`2`hi`'
    # input_data = train('train.txt')
    last_output_data = None
    for data in input_data:
        data_binary_str = toBin(data)
        
        # log(data, 'input_data')
        # log(data_binary_str, 'input_data_binary')

        output_binary_strs, weight = PROCESSOR.process(data_binary_str)
        od = []
        output_data = None    
        for output_binary_str in output_binary_strs:
            output_data = toChar(output_binary_str)

            # log(output_binary_str, 'output_data_binary')
            # log(output_data, 'output_data')
            od.append(output_data)
        last_output_data = ', '.join(od)
        print('input_data = {}, predicted_output_data = {}, weight = {}'.format(data, last_output_data, weight))

if __name__ == '__main__':
    main()
    