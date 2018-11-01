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

def main():
    # start the sensor
    input_data = '''hello`hi`i live in nigeria, it is a very big place`hello`hi`what is 1+4?`hello`hi`my name is jack`hi`i am james, i come from the north.`do you speak latin?`no`good morning`hi`how are you doing?`i am great thank you`hello, my name is uri`hi, i'm emma.`hey`hey`can you come today?`no`see me tommorow`hi janet`hello kareem`hello`'''

    # input_data = "hello`hi`hello`hi`"
    # input_data = 'hi`hello`1+1 is 2`1+4 is 5`2+3 is 5`5+2 is 7`3+1 is 4`3+4 is 7`2+1 is 3`4+4 is 8`what is 2+3?`5`what is 2+1?`4`what is 3+4?`7`what is 5+2?`7`what is 1+1?`2`hi`'
    for data in input_data:
        output = PROCESSOR.process(data)
        print('input = {}, outputs = {}'.format(data, output))


if __name__ == '__main__':
    main()