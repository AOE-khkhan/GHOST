
import cv2, os, PIL

'''
explorer00000001 - play
'''
videos = {'video':[]}
images = {'image':['png', 'jpeg', 'jpg']}

all_media = [videos, images]
media_types = {z:y for x in all_media for y in x for z in x[y]}

'''
TabbedPanel
============

Test of the widget TabbedPanel.
'''

from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.lang import Builder

Builder.load_string("""

<Test>:
    size_hint: .5, .5
    pos_hint: {'center_x': .5, 'center_y': .5}
    do_default_tab: False

    TabbedPanelItem:
        text: 'first tab'
        Label:
            text: 'First tab content area'
    TabbedPanelItem:
        text: 'tab2'
        BoxLayout:
            Label:
                text: 'Second tab content area'
            Button:
                text: 'Button that does nothing'
    TabbedPanelItem:
        text: 'tab3'
        RstDocument:
            text:
                '\\n'.join(("Hello world", "-----------",
                "You are in the third tab."))

""")


class Test(TabbedPanel):
    pass


class TabbedPanelApp(App):
    def build(self):
        return Test()

class File:
    def __init__(self, file):
        self.file = file
        self.name = self.file.split("\\")[-1]
        if len(self.file.split(".")) > 1:
            self.extension =  self.file.split(".")[-1]
            self.dir = "\\".join(self.file.split("\\")[:-2]) + '\\' + self.file.split("\\")[-2].split("\\")[-1]
        
        else:
            self.extension =  ''
            self.dir = self.file.split("\\")[-1].split("\\")[-1]
        
        self.path = self.file

    
class explorer:
    def run_gui(self, file):
        TabbedPanelApp().run()
        file = File(file)
        self.files = self.get_all_files(file.dir)
        print(file.dir, self.files)
        if len(self.files) > 0:
            if file.extension == '':
                pass
            else:
                self.i = self.files.index(file.name)
            media_type = self.get_mediaType(file.extension)
            self.explorer00000001(file, media_type)

    def explorer00000001(self, file, media_type):
        if media_type == 'image':
            img = cv2.imread(file.path)
            cv2.startWindowThread()
            cv2.imshow(file.name, img)
            self.focus = 'explorer00000001'
            
        else:
            return False
        
    def get_mediaType(self, extension):
        if extension in media_types:
            return media_types[extension]
        else:
            return False

    def get_all_files(self, path):
        fl = []
        for r in os.walk(path):
            for d in r:
                fl.append(d)
        return fl[2]
    
##    media = file('C:\\Users\\christian\\Pictures\\MillieBobbyBrownSigmaBirdyVideo.png')
    

##if __name__ == '__main__':
    
