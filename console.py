
class console:
    def getInput(self):
        
        if self.switch is True:
            getInput = input("\_> ")
        else:
            getInput = input("%s_> "%(self.char))
            if getInput.startswith(">") and len(getInput) > 1:
                getInput = getInput[1:]
            else:
                getInput = self.char+getInput
        
        if getInput in ["??","?","?weight","?relation","?memory","?learn","?read","?eye","?mouth"]:
            self.switch = False
            self.char = getInput
            print()
            #print("\n##############terminal mode(%s)##############\n"%(self.char[1:]))
            
            return ""
        
        elif getInput in ["??-","?weight-","?relation-","?memory-","?learn-","?read-","?eye-","?mouth-"]:
            self.char = "?"
            self.switch = False
            print()
            #print("\n##############terminal mode(%s)##############\n"%(self.char[1:]))
            
            return ""
        
        elif getInput in ["??>","?>","?weight>","?relation>","?memory>","?learn>","?read>","?eye>","?mouth>"]:
            print()
            #print("\n##############ghost mode##############\n")
            self.switch = True

            return ""

        else:
            return getInput

    def switchSource(self):
        self.lastSource = self.source
        if self.source == "x":
            self.source = "y"
        else:
            self.source = "x"
