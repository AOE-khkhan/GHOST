def do_nothing(*args):
	pass

class Console:
	def __init__(self):
		self.log = print

	def logState(self, state=None):
		if state == None:
			if self.log_state == False:
				self.log_state = True

			else:
				self.log_state = False
		else:
			self.log_state = state

		if self.log_state:
			self.log = print

		else:
			self.log = do_nothing

		return

	def setLogState(self, state):
		return self.logState(state)

	def toggleLogState(self):
		return self.logState(None)
