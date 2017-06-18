import os
class terminal:
    def runCommand(self,cmd):
        if cmd.startswith("??"):
            st = cmd.replace("??","").strip()
##            try:
            r = eval(st)
            if r == None or (type(r) == str and len(r) < 1):
                return "Done"
            else:
                return str(r)
##            except Exception as e:
##                return "Error: "+str(e)
        elif cmd.startswith("?rollback"):
            st = cmd.replace("?rollback","").strip()
            if st == "":
                i = -1
            else:
                try:
                    i = int(st)
                except Exception as e:
                    return "Error: Invalid Syntax, require int."
            
            dirs = []
            for r, d, f in os.walk(os.getcwd()+'\\sessions\\'):
                dirs.append(d)

            dirs = sorted(dirs[0])
            if abs(i) > len(dirs)-1:
                return "Error: Invalid index, out of range({} to {}).".format(0, len(dirs)-1)
            else:
                print("rolling back...")
                os.system('robocopy "'+os.getcwd()+'\\sessions\\'+dirs[i]+'" "'+os.getcwd()+'" /E')
                self.loadMemory()
                return "Done!"
                
        elif cmd == "?cachestate":
            dirs = []
            for r, d, f in os.walk(os.getcwd()+'\\sessions\\'):
                dirs.append(d)

            dirs = sorted(dirs[0])
            print("saving GHOST (memory)state...")
            m = str(int(dirs[-1])+1)
            os.mkdir('sessions/'+m)
            os.mkdir('sessions/'+m+'/memory')
            os.system('robocopy "'+os.getcwd()+'\\memory" "'+os.getcwd()+'\\sessions\\'+m+'\\memory" /E')
            self.loadMemory()
            return "Done!"
                
        else:
            return "Error: Unknown Command"
