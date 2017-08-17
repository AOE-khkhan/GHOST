import time
from functions import *
from collections import Counter

from train import trainingData, codebaseTrainingData
from console import console
from brain import intelligence
from mouth import voice

class bot(console, intelligence, voice):
    ghostName = "Iris"
    session = []
    context = []
    tpoint = []
    memory = []
    codebase = {}
    rev_codebase = {}
    events = {}
    expected_ans = []
    new_expected_ans = []
    events_id = {}
    freq = []
    show_info = True
    confirmed_event = False
    reply = False
    reply_value = ''
    predicted_list = []
    last_predicted_list = []
    
    source = lastSource = ""
    
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
        if self.test_state == 0:
            self.createMemory()
            self.train()
            
        else:
            self.loadMemory()

        while self.state is True:
            #initializations
            self.confirmed_event = False
            
            #using the console class to grab input from user
            self.string = self.getInput()
            
            #pushing recieved string to brain for analysis
            self.output()
            
            for x in self.new_expected_ans: self.set_expected_ans(x, self.expected_ans)
            self.new_expected_ans = []        

    def output(self):
        if len(self.string) != 0:
            r = self.process(self.string)
            if self.string.strip() != '' or r is not None:
                if type(r) == tuple and r[0] != None and r[-1] == 1:
                    r = r[0]
                    self.reply = True
                    self.reply_value = r
                    self.setReply(self.string, r)
                    self.last_predicted_list = self.predicted_list
                    self.predicted_list = []
                    if self.speak_state:
                        self.speak(r)
                else:
                    self.reply = False
                    r = ''
                print('\n{}\n'.format(r))
            
    def process(self,data):
        return self.analyse(data)

    def trainCodebase(self):
        c = 0
        #self.chunk_learn(filename)
        self.learning = 1
        for td in codebaseTrainingData:
            if len(td[0]) > 0:
                c += 1
                
                #save the object name
                self.saveMethod(td[0], td[1])
                
                print("training ghost codebase {}% complete".format(formatVal((c/len(codebaseTrainingData))*100)))
        self.learning = 0
        self.loadCodebase()
        
    def train(self):
        c = 0
        #self.chunk_learn(filename)
        self.trainCodebase()
        self.learning = 1
        for td in trainingData:
            if len(td[0]) > 0:
                c += 1
                
                #save the object name
                if td[0] not in self.memory:
                    self.saveInput(td[0])
                    
                self.setContext(td[0])
                self.setReply(td[0], td[1]) #save obj property
                self.setContext(td[1])
                print("training ghost {}% complete".format(formatVal((c/len(trainingData))*100)))
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

    def chunk_learn(self, filename, common=100):
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
