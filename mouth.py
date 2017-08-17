import sys, os
from threading import Thread
import subprocess

speak_state = True

class voice:
    def call(self):
        subprocess.check_output("python speak.py", shell=True)
                                       
    def speak(self, string):
        self.write_to_speak(string)
        t1 = Thread(target=self.call, args=())
        t1.start()
        return 'xxx{[s00000001]}xxx'
    
    def stop(self):
        self.speak('')
        return 'xxx{[s00000000]}xxx'
        
    def write_to_speak(self, string):
        file = open('resources\\text.txt', 'w')
        file.write(string)
        file.close()
