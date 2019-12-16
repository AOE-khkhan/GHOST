
# import lib code
from context import Context
from console import Console
from memory_line import MemoryLine

class KeyboardProcessor(object):
	''' Keyborad Processor file'''
	def __init__(self, cortex):
		# initialize console
		self.console = Console()
		self.log = self.console.log

		self.console.setLogState(True)

		# the central processor object
		self.cortex = cortex
		self.cortex.keyboard_processor = self

		# the history
		context_maxlength = 10
		self.context = Context(context_maxlength)

		# initialize magnetic_memory_strip
		self.keyboard_memory_line = MemoryLine()


	def run(self, keystroke, verbose=0):
		self.keyboard_memory_line.add(keystroke, allow_duplicate=True)

