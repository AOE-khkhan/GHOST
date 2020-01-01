# from std lib
import os

# from thrid party lib
import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.optimize import minimize

# key identifiers for memory
TIMESTAMP = '_timestamp_'


def optimize(X, Y, verbose=1):
    def mse(matrix):
        return np.sqrt((matrix**2).mean())

    def transform1(x):
        return np.array(x) * X

    def transform(x):
        xcoord, ycoord = X[:, 0], X[:, 1]
        return np.array([(xcoord * x[0]) + x[1], (ycoord * x[2]) + x[3]]).T

    def constraint(x):
        return mse(Y - transform(x))

    def objective(x):
        return mse(np.array(x))
#         return mse(np.array(x)) + (Y - transform(x)).sum()

    def callback(x):
        solutions.append(x)

    solutions = []
    max_guess = 100
    n_params = 4

    # initial guesses
    # x0=np.random.randint(max_guess, size=total_number_of_options)
    x0 = np.full(n_params, max_guess)

    # solutions
    solutions = [x0]

    # the bounds
    b = (-max_guess, max_guess)
    bnds = tuple(b for _ in range(n_params))

    # define constriants
    cons = ([
        {'type': 'eq', 'fun': constraint},
    ])

    solution = minimize(objective, x0, method='SLSQP', bounds=bnds,
                        constraints=cons, callback=callback)
    x = solution.x

    # show initial objective
    if verbose:
        print(X.shape, Y.shape)
        print('Initial SSE Objective: {:.4f}, x0 = {}, error = {:.4f}'.format(
            objective(x0), x0, constraint(x0)
        )
        )

        # show final objective
        print('Final SSE Objective: {:.4f}, x = {}, error = {:.4f}'.format(
            objective(x), x, constraint(x)
        ), end='\n\n'
        )

    return objective(x), constraint(x)


def load_image(image_path):
    image = cv2.imread(image_path)
    image = image[..., ::-1]
    image = np.array(image, dtype=np.int64)
    return image


def resultant(matrix):
    return round(np.sum(matrix), 4)


def get_similarity_ratio(a, b):
    return (255**-1) - abs(a - b).mean()


def toGrey(image):
    return image.mean(2)


def index_row_in_array(row, arr):
    return np.where((row == arr).all(tuple(range(len(arr.shape)))[1:]) == True)[0]


def is_row_in_array(row, arr):
    return len(index_row_in_array(row, arr)) != 0


def validateFolderPath(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    return labeled_img
