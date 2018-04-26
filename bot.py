import time, json
from functions import *
from collections import Counter

from train import trainingData
from console import Console
from brain import Intelligence
##from mouth import voice
##from GUI import GUI

r = time.time()
def tt(string=''):
    global r
    x = time.time()
    print('{} in {} seconds'.format(string, round(x-r, 4)))
    r = x
    
class bot(Console, Intelligence):#, GUI):
    PROPERTIES = {"ifreq":0, "freq":0, "ans":{}}
    MEMORY_PATH = "memory"
    CONSOLE_MEMORY_PATH = MEMORY_PATH+"/console/"
    BLOCK_SIZE = 256
    INDEX_SIZE = 1024
    
    ghostName = "Iris"
    string = ''

    #Session data(from session 1 to other)
    xxque, xxans, xxqna_infl, xxque_sessions, xxans_sessions, xxin_memory = [], [], [], [], [], []

    context = []
    
    cache_data = {}
    
    expected_ans = []
    new_expected_ans = []

    show_info = True
    confirmed_event = False
    reply = False

    reply_value = ''

    predicted_list = []
    last_predicted_list = []

    used_event = False          #for ans
    
    source = "x"
    lastSource = ""
    cui = "home"

    lastInputSource = ""
    
    #for console input in user mode
    switch = True
    
    #ghost test state(default is on for test)
    test_state = 1
    
    learning = 0

    #ghost state(default is alive==1)
    state = True

    show_process_state = True
    #ghost speak state
    speak_state = 0

    def __init__(self):
        tt('start')
        if self.test_state == 0:
            self.createMemory()
            self.train()
            
        else:
            self.loadMemory()
        tt('memory set up')

        while self.state is True:
            #initializations
            self.confirmed_event = False

            #using the console class to grab input from user
            while len(self.string.strip()) < 1:
                self.string = self.getInput()
            tt('grabbed user input')
            
            self.string = self.string.strip()
            #pushing recieved string to brain for analysis
            if self.string != '':
                self.switchSource()
                self.output(self.string)
                self.switchSource()

            tt('output')
            self.string = ""
            
    def output(self, string, callback=False):
        r = self.process(string)
        tt('process')
        
        if r is not None:
            if type(r) == tuple:
                r, callback = r

            if self.speak_state:
                self.speak(r)
            
        else:
            r = ''
        print('\n{}\n'.format(r))
        
        if callback:
            self.output(r, callback)

    def process(self,data):
        return self.analyseInput(data)
    
    def train(self):
        c = 0
        self.learning = 1
        self.source = "x"
        length = len(trainingData)
        for td in trainingData:
            c += 1
            if len(td[0]) > 0:
                
                #log training data
                td_index = self.save2memory(td[0])
                self.log(td[0], td_index)
                
            if len(td[1].strip()) > 0:
                self.switchSource()
                
                td_index = self.save2memory(td[1])
                self.log(td[1], td_index)

                self.switchSource()

            print("training ghost {}% complete".format(formatVal((c/length)*100)))
        self.learning = 0
        
    def learn(self,filename):
        filename = 'corpus/'+str(filename)
        file = open(filename+".txt","r")
        r = file.read()
        file.close()

        r = r.replace("\n","")
        trainingData = sent_tokenize(r)
        
        c = 0
        self.chunk_learn(filename)
        self.learning = 1
        for td in trainingData:
            td = td.strip()
            if len(td) > 0:
                c += 1
                #save the object name
                if td not in self.memory:
                    self.saveInput(td)
                    
                self.setContext(td)
                print("ghost reading '{}': {}% complete".format(filename,formatVal((c/len(trainingData))*100)))
        self.learning = 0

    def chunkLearn(self, filename, common=100):
        files = [open(str(filename)+".txt").read()]
        words = []
        for xx in files:
            words.extend(xx.split())
        words = Counter(words)
        words = words.most_common(common)
        c = 0
        self.learning = 1
        for w in words:
            c += 1
            #save the object name
            if w[0] not in self.memory:
                self.saveInput(w[0])
                
            self.setContext(w[0])
            print("ghost learning '{}' chunks: {}% complete".format(filename,formatVal((c/len(words))*100)))
        self.learning = 0

    def test(self, input_list, output_list):
        for n in range(len(input_list)):
            print('testing',input_list[n])
            r = self.process(str(input_list[n]))
            print(r)
            if r == str(output_list[n]):
                print( '-----------------------------!!!CORRECT!!!----------------------------')
            else:
                print( '-----------------------------!!!WRONG!!!----------------------------')

if __name__ == "__main__":
    IRIS = bot()
##self.test(['2 + 4', '3 + 4', '5 + 6', '3 + 9', '5 - 4', '8 - 2', '3 - 1', 'subtract 2 from 8', 'subtract 5 from 10', 'subtract 3 from 9'], ['6','7', '11', '12', '1', '6', '2', '6', '5', '6'])
