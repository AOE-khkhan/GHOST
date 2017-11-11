import sys, os
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup

import os
print(sys.argv)
class Root(FloatLayout):
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)

    def __init__(self):
        self.load()
        
    def load(self, filename=os.getcwd()+"\\"+"".join(sys.argv[1:])):
        print(filename)
        with open(filename, 'r') as stream:
            self.text = stream.read()

    def save(self, path, filename):
        with open(os.path.join(path, filename), 'w') as stream:
            stream.write(self.text)


class Editor(App):
    pass

Factory.register('Root', cls=Root)

if __name__ == '__main__':
    Editor().run()
