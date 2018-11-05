'''
Author: Joshua, Christian r0b0tx
Date: 1 Nov 2018
File: processor.py
Project: GHOST
'''


# import from python standard lib
import re
from itertools import combinations

class Processor:
    """docstring for Processor"""    
    
    def __init__(self):
        self.log_state = True

    def log(self, output, title=None):
    	if not self.log_state:
    		return

    	if title != None:
    		print('\n {} \n{}\n'.format(title, ''.join(['=' for _ in range(len(title)+2)])))

    	print(output)
    	return

    def getDendrites(self, input_list):
    	return [i for i, x in enumerate(input_list) if x == '1']

    def getHighOrderDendrites(self, dendrites):
    	'''
    	this combines dendrites to see which combination influences next process
    	'''
    	length = len(dendrites)
    	new_dendrites = []
    	for r in range(2, length):
    		combs = combinations(dendrites, r)

    		for comb in combs:
    			new_dendrites.append(comb)

    	new_dendrites.extend(dendrites)
    	return new_dendrites

    def process(self, binary_str):
    	input_list = list(binary_str)

    	# get the dendrites formed
    	dendrites = self.getDendrites(input_list)

    	self.log(dendrites, 'dendrites')

    	# get high order dendrites
    	dendrites = self.getHighOrderDendrites(dendrites)

    	self.log(dendrites, 'dendrites updated')

    	return