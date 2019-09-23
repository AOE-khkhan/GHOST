class Context:
	def __init__(self, maxlength=10):
		self.maxlength = maxlength
		self.context = []

	def append(self, index):
		self.context.append(index)
		if len(self.context) == self.maxlength:
			self.context.pop(0)
		return