# imprt from python std lib
import time

def active_timer(func):
	# the timer starts
	start = time.time()

	# the function is called
	result = func()

	# the time taken for process
	time_taken = round(time.time() - start, 4)

	# the time result
	print('{:>8}sec(s) for {}'.format(time_taken, func.__qualname__), end="\n\n")

	# the return:result form function call
	return result

def inactive_timer(func):
	return func()

class Timer:
	def __init__(self, state=True):
		self.timer_state = state
		self.TimerState()

	def TimerState(self):
		if self.timer_state:
			self.run = active_timer

		else:
			self.run = inactive_timer

		return

	def setTimerState(self, state):
		self.timer_state = state
		self.TimerState()

	def toggleTimerState(self):
		if self.timer_state == False:
			self.timer_state = True

		else:
			self.timer_state = False

		self.TimerState()
