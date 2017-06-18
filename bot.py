import time
from functions import *
from collections import Counter

from train import trainingData, codebaseTrainingData
from console import console
from brain import intelligence

class bot(console, intelligence):

    ghostName = "Iris"
    session = []
    context = []
    tpoint = []
    memory = []
    codebase = []
    freq = []
    
    source = lastSource = ""
    
    #for console input in user mode
    switch = True
    
    #ghost test state(default is on for test)
    test = 1
    
    learning = 0

    #ghost state(default is alive==1)
    state = True

    show_process_state = True
    
    #ghost speak state
    speak_state = 0

    def __init__(self):
        if self.test == 0:
            self.createMemory()
            self.train()
            
        else:
            self.loadMemory()

        while self.state is True:
            #using the console class to grab input from user
            self.string = self.getInput()
            
            #pushing recieved string to brain for analysis
            if len(self.string) != 0:
                print()
                r = self.process(self.string)
                if r is not None:
                    print(r)
                    #self.speak(r,self.speak_state)
                    print()


    def createMemory(self):
        self.create("memory")
        self.create("codebase")
        self.create("session")
        self.create("context")
        self.create("freq")
        self.save()
        
        
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
                self.saveMethod(td[0], td[1]+"-----"+td[2])
                
                print("training ghost codebase {}% complete".format(formatVal((c/len(codebaseTrainingData))*100)))
        self.learning = 0
        
    def train(self):
        c = 0
        self.trainCodebase()
        self.learning = 1
        for td in trainingData:
            if len(td[0]) > 0:
                c += 1

                self.process(td[0])
                self.setReply(td[0], td[1]) #save obj property
                self.process(td[1])
                print("training ghost {}% complete".format(formatVal((c/len(trainingData))*100)))
        self.learning = 0
        
    def learn(self,filename):
        file = open(str(filename)+".txt","r")
        r = file.read()
        file.close()

        r = r.replace("\n","")
        trainingData = sent_tokenize(r)
        
        c = 0
        self.learning = 1
        for td in trainingData:
            td = td.strip()
            if len(td) > 0:
                c += 1
                #save the object name
                self.process(td)
                self.setContext(td)
                print("ghost reading '{}': {}% complete".format(filename,formatVal((c/len(trainingData))*100)))
        self.learning = 0

    def create(self, name):
        file = open("memory/console/"+name+".txt","w")
        file.write("")
        file.close()

if __name__ == "__main__":
    IRIS = bot()
