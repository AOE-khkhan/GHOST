from random import randint

# from Library code
from probability_model import ProbabilityModel
from utils import simulate_addition, simulate_count, transform

# from thrid pirty
from tqdm import tqdm

def main():
    # intials
    number_of_iterations = 71

    # the data set
    x_train = ''.join([simulate_count(1, end) for end in range(10, number_of_iterations, 10)])
    x_test = simulate_count(71, 81)

    # initialize the ProbabilityGraph
    probability_model = ProbabilityModel(context_size=3, models_garbage_batch=8)

    # ------------------------- train the ProbabilityGraph ---------------------------------------
    for character in tqdm(x_train):
        # character as an integer
        input_value = transform(character)

        # pass current context into ProbabilityGraph to predict the possible outcome
        probability_model.run(input_value)

    # ------------------------- test the ProbabilityGraph ---------------------------------------
    for index, character in enumerate(x_test[:-1]):
        # character as an integer
        input_value = transform(character)

        # pass current context into ProbabilityGraph to predict the possible outcome
        prediction, confidence = probability_model.run(input_value)

        note = '*' if confidence < .5 or prediction != transform(x_test[index+1]) else ''

        # display the info
        print(
            ''.join(list(map(transform, probability_model.context_list))),
            transform(prediction) if prediction is not None else '%',
            f"{confidence:.4f} {note}",
            sep=' -> '
        )

    probability_model.save()


if __name__ == "__main__":
    main()
