import os, json

from functions import *

GUI_PATH = "GUI\\"
GUI_RESOURCES_PATH = "resources\\UI\\"
UIs = read_json(GUI_RESOURCES_PATH+"UIs")


class GUI:
    cui = "home"
    
    def load_gui(self):
        if self.cui in UIs:
            self.ui = UI(self.cui)
        
    def setProperty(self, name, value):
        self.ui.properties[name] = value

    def render_gui(self):
        args = [x+"='"+self.ui.properties[x]+"'," for x in self.ui.properties.keys()]
        args = "".join(args)
        if args[-1] == ",":
            args = args[:-1]
##        print(type(self.ui.format), self.ui.format[:-30])

        gui = eval("self.ui.format.format("+args+")")
        print(len(gui))
        gui = self.ui.pre_format + gui
        print(len(gui))
        
        print(gui[:10])
        write_file(GUI_PATH+"\\view.html", gui)


class UI:
    def __init__(self, name):
        self.name = name
        self.properties = read_json(GUI_RESOURCES_PATH + name + "\\" + name)
        self.load_UI()
        
    def load_UI(self):
        self.format = self.pre_format = ''
        for x in UIs[self.name]["dependencies"]:
            if x.endswith("js"):
                self.pre_format += "<script>"
            self.pre_format += self.open(GUI_PATH+x)
            if x.endswith("js"):
                self.pre_format += "</script>"
        self.format = self.open(GUI_RESOURCES_PATH + self.name + "\\" + self.name + ".txt")
        
    def open(self, name):
        file = open(name, "r")
        r =  file.read()
        file.close()
        return r
