import json

class Transformer:
    def __init__(self, offset=0, numbers_path='prime_numbers_list.json'):
        #reserve some idx for other tasks
        self.OFFSET = offset

        # load all the numbers to map to
        self.load_numbers(numbers_path)

    def load_numbers(self, numbers_path):
        ''' retrive all the prime numbers '''
        # load prime_numbers as a json object
        with open(numbers_path) as file:
            self.PRIME_NUMBERS = tuple(json.load(file))

    def transform(self, index):
        ''' transform the index to idx in prime numbers list '''
        if type(index) == str:
            index = ord(index)

        return self.PRIME_NUMBERS[self.OFFSET + index]

    def reverse(self, idx, to_chr=True):
        ''' collect idx derived from prime numbers list '''
        index = self.PRIME_NUMBERS.index(idx) - self.OFFSET

        if to_chr:
            return chr(index)
        return index
