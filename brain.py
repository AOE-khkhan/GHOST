import time

from collections import Counter

from subsetter import Subsetter
from terminal import Terminal
from brain_functions import BrainFunctions
from processor import Processor
from functions import *

r = time.time()
def tt(string=''):
    global r
    x = time.time()
    print('{} in {} seconds'.format(string, round(x-r, 4)))
    r = x

class Intelligence(Subsetter, Terminal, BrainFunctions, Processor):
    def analyseInput(self, data):
##        checks if input is a command or huamn language
        r = self.prepareInput(data)
        tt('prepared data')
        
        if r == 'prepared':
            return self.processInput(data)
        
        else:
            return r
    
    def prepareInput(self, data):
        if data.startswith("?"):
            print(self.runCommand(data))
            tt('ran command')
        else:
            if data not in ["~","/"]:
                data_index = self.save2memory(data)
                tt('saved to memory')
                
            self.log(data, data_index)
            tt('logged data')
            
            if data not in ["~","/"]:
                self.expected_ans = []
                
            if data == "~":
                self.switchSource()
                tt('switched and managed source')
                
            return 'prepared'

    def processInput(self, data):
        if data not in ["~","/"]:
            if not self.learning:
                #to compare present data with previous data and lock relation
                return self.runInputProcessing(data)
                              
        
    def save2memory(self, data, artificial=0):
        check = self.dataInMemory(data)
        if check:
            if artificial == 1:
                check = self.setMemoryData(data, "ifreq", int(self.getMemoryData(data, "ifreq")) + 1)

            else:
                check = self.setMemoryData(data, "freq", int(self.getMemoryData(data, "freq")) + 1)
                
        else:
            if artificial == 1:
                check = self.setMemoryData(data, "freq", 1)
                check = self.setMemoryData(data, "ifreq", 0)

            else:
                check = self.setMemoryData(data, "ifreq", 1)
                check = self.setMemoryData(data, "freq", 0)
        
        return check

    def toogleState(self, var):
        if var:
            var = False
            
        else:
            var = True

    
