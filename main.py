from random import randint

# from Library code
from probability_model import ProbabilityModel
from utils import save, transform, mean, SimulateGridWorld

# from thrid pirty
from tqdm import tqdm

def main():
    grid_world = SimulateGridWorld(5, space_token='-')
    callback = lambda self: list(map(transform, self.world.ravel()))

    # intials
    number_of_iterations = 500
    movements = ['r', 'l', 'z']

    # the data set
    x_train = ''.join([movements[randint(0, 1)] for _ in range(number_of_iterations)])
    x_test = 'rrr'

    # initialize the ProbabilityGraph
    probability_models = [
        ProbabilityModel(idx=idx, space_batch_size=6, time_batch_size=1, models_garbage_batch=8) for idx in range(6)
    ]

    # ------------------------- train the ProbabilityGraph ---------------------------------------
    for character in tqdm(x_train):
        # next input from sensors
        current_state = callback(grid_world) + [transform(character)]

        # pass current context into ProbabilityGraph to predict the possible outcome
        for probability_model in probability_models:
            probability_model.run(current_state)

        # make a move in the grid world
        grid_world.move(character, callback)

    # ------------------------- test the ProbabilityGraph ---------------------------------------
    for index, character in enumerate(x_test[:-1]):
        # current input from sensors
        current_state = callback(grid_world) + [transform(character)]
        
        # make the move in the world
        next_character = x_test[index+1]
        future_state = grid_world.move(character, callback) + [transform(next_character)]

        final_prediction, final_actual = '', ''

        # pass current context into ProbabilityGraph to predict the possible outcome
        for idx, probability_model in enumerate(probability_models):
            prediction, confidence = probability_model.run(current_state)
            context = ''.join(list(map(transform, probability_model.context_list)))
            note = '*' if confidence < .5 or prediction != future_state[idx] else ''

            prediction = transform(prediction) if prediction is not None else '%'
            final_prediction += prediction

            actual = transform(future_state[idx])
            final_actual += actual

            # display the info
            print(f'  [{context}]: [{transform(current_state[idx])}] => [{actual}], predicted => [{prediction}] with {confidence*100:.2f}% {note}')

        # aggregate the information for full output
        note = '*' if final_prediction != final_actual else ''
        print(f'[{context}] => [{final_actual}], predicted => [{final_prediction}] {note}')

    # keep all models for all sensors
    save(probability_models)


if __name__ == "__main__":
    main()
