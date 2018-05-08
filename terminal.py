import os
class Terminal:
    def runCommand(self,cmd):
        if cmd.startswith("??rollback "):
            st = cmd.replace("??rollback ","")

            os.system('del memory\\console\\events\\ /q')

            os.system('ROBOCOPY "sessions\\'+st+'" "memory" /E')
            self.loadMemory()
            
        elif cmd.startswith("??cache "):
            st = cmd.replace("??cache ","")
            os.system('mkdir "sessions\\'+st+'"')
            os.system('del "sessions\\'+st+'\\console\\events\\" /q')
            os.system('ROBOCOPY "memory" "sessions\\'+st+'" /E')
        
        elif cmd.startswith("??quit"):
            self.STATE = False
            
        elif cmd.startswith("??open "):
            self.setProperty("class_name", cmd.replace("??open ",""))
            self.render_gui()
            
        elif cmd.startswith("??"):
            st = cmd.replace("??","")
            # r = eval(st)
            try:
                r = eval(st)
                if r == None or (type(r) == str and len(r) < 1):
                    return "Done"
                
                else:
                    return str(r)
                
            except Exception as e:
                return "Error: "+str(e), None
            
        else:
            return "Error: Unknown Command"
