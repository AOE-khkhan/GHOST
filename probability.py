class Probability:
    def __init__(self, n=0, N=0):
        self.n = n
        self.N = N

    def record(self, value):
        ''' record an observation '''
        self.n += value
        self.N += value if value else 1
    
    def get_value(self):
        ''' the probability value as a fraction '''
        return self.n/self.N if self.N else 0
