'''
Explorer example
=============

This example plays sounds of different formats. You should see a grid of
buttons labelled with filenames. Clicking on the buttons will play, or
restart, each sound. Not all sound formats will play on all platforms.

All the sounds are from the http://woolyss.com/chipmusic-samples.php
"THE FREESOUND PROJECT", Under Creative Commons Sampling Plus 1.0 License.

'''

'''
explorer00000001 - play
'''
videos = {'video':['mp4', 'mkv']}
images = {'image':['png', 'jpeg', 'jpg']}
text = {'text':['txt', 'py', 'php', 'js', 'css', 'html', 'cshtml']}

all_media = [videos, images, text]
media_types = {z:y for x in all_media for y in x for z in x[y]}

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


import kivy
kivy.require('1.0.8')

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.core.audio import SoundLoader
from kivy.properties import StringProperty, ObjectProperty, NumericProperty
from glob import glob
from os.path import dirname, join, basename


class ExplorerButton(Button):

    filename = StringProperty(None)
    sound = ObjectProperty(None, allownone=True)
    volume = NumericProperty(1.0)

    def on_press(self):
        print('filename', self.filename)
        file = File(self.filename)
        media_type = self.get_mediaType(file.extension)
        self.explorer00000001(file, media_type)

    def explorer00000001(self, file, media_type):
        print('opening {} file: name= {}, path = {}'.format(media_type, file.name, file.path))

        
    def get_mediaType(self, extension):
        if extension == '':
            return 'folder'
        
        elif extension in media_types:
            return media_types[extension]
        
        else:
            return False

class ExplorerBackground(BoxLayout):
    pass


class ExplorerApp(App):
    
    def build(self):
        root = ExplorerBackground()
        for fn in glob(join(dirname(__file__), '*.*')):
            btn = ExplorerButton(
                text=basename(fn).replace('_', ' '), filename=fn,
                size_hint=(None, None), halign='center',
                size=(1000, 30), text_size=(50, None))
            root.ids.sl.add_widget(btn)

        return root

    def release_audio(self):
        for audiobutton in self.root.ids.sl.children:
            audiobutton.release_audio()

    def set_volume(self, value):
        for audiobutton in self.root.ids.sl.children:
            audiobutton.set_volume(value)


if __name__ == '__main__':
    ExplorerApp().run()
