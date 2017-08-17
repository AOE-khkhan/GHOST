import pyttsx
import sys, os

filename = " ".join(sys.argv[1:])

def read_text():
    global filename
    if len(filename) < 1: filename = 'resources\\text.txt'
    if os.path.exists(filename):
        text_file = open(filename)
        text = text_file.read()
        text_file.close()
    else:
        text = filename
    return text

def onWord(name, location, length):
    new_text = read_text()
    if text != new_text:
        engine.stop()
        os.system("python speak.py resources\\text.txt")
        
engine = pyttsx.init()
engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0')
text = read_text()
engine.connect('started-word', onWord)
engine.say(text)
engine.runAndWait()

