from random import randint
from probability_model import ProbabilityModel

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
    start, end = 21, 31
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
    probability_model = ProbabilityModel(context_size=8, models_garbage_batch=8)

    # ------------------------- train the ProbabilityGraph ---------------------------------------
    for character in tqdm(x_train):
        # character as an integer
        input_value = ord(character)

        # pass current context into ProbabilityGraph to predict the possible outcome
        probability_model.run(input_value)

    # ------------------------- test the ProbabilityGraph ---------------------------------------
    for index, character in enumerate(x_test[:-1]):
        # character as an integer
        input_value = ord(character)

        # pass current context into ProbabilityGraph to predict the possible outcome
        prediction, confidence = probability_model.run(input_value)

        note = '*' if confidence < .5 or prediction != ord(x_test[index+1]) else ''

        # display the info
        print(
            ''.join(list(map(chr, probability_model.context_list))),
            chr(prediction) if prediction is not None else '%',
            f"{confidence:.4f} {note}",
            sep=' -> '
        )

    probability_model.save()


if __name__ == "__main__":
    main()
