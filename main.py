
# import lib code
from processor import Processor

# initialize objects
PROCESSOR = Processor()

def main():
    # start the sensor
    # input_data = "hi`hi`what is 1+4?`hello1`hi`my name is jack`hi`"
    input_data = "hi`hello`hi`"
    for data in input_data:
        output = PROCESSOR.process(data)
        print(output)


if __name__ == '__main__':
    main()