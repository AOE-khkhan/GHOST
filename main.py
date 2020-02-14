from random import randint
from probability_network import ProbabilityNetwork

from tqdm import tqdm

def simulate_addition(start, stop):
    text = ''
    for num1 in range(start, stop+1):
        for num2 in range(start, stop+1):
            text += f'{num1}+{num2}~{num1+num2}~'
    return text


def simulate_count(start, stop):
    text = f'count_{start}_to_{stop}~'
    for number in range(start, stop+1):
        text += f'{number}~'
    return text


def main():
    # intials
    start, end = 1, 101
    number_of_iterations = 10

    # the data set
    x_train = ''.join(
        [simulate_count(1, randint(start, end)) for _ in range(number_of_iterations)]
        + [simulate_count(randint(start, end), randint(start, end)) for _ in range(number_of_iterations)]
        + [simulate_addition(1, 10) for _ in range(number_of_iterations)]
    )
    x_test = simulate_count(1, 10)
    # x_test = simulate_addition(1, 2)

    # initialize the ProbabilityGraph
    probability_network = ProbabilityNetwork(3)

    # ------------------------- train the ProbabilityGraph ---------------------------------------
    for character in tqdm(x_train):
        # pass current context into ProbabilityGraph to predict the possible outcome
        prediction = probability_network.run(character)

    # ------------------------- test the ProbabilityGraph ---------------------------------------
    for index, character in enumerate(x_test[:-1]):
        # pass current context into ProbabilityGraph to predict the possible outcome
        prediction = probability_network.run(character, 1)

        # display the info
        print(probability_network.context_manager.context, x_test[index+1], prediction, sep=' -> ')

    # probability_network.save()


if __name__ == "__main__":
    main()
