def do_nothing(*args):
	pass

class Console:
	def __init__(self, state=True):
		self.log_state = state
		self.logState()

	def logState(self):
		if self.log_state:
			self.log = print

		else:
			self.log = do_nothing

		return

	def setLogState(self, state):
		self.log_state = state
		self.logState()

	def toggleLogState(self):
		if self.log_state == False:
			self.log_state = True

		else:
			self.log_state = False
		self.logState()