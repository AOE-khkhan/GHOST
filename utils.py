# from std lib
import os

# from thrid party lib
import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided


def getKernels(img, kernel_size):
	return as_strided(
		img,
		shape=(
			img.shape[0] - kernel_size[0] + 1,  # The feature map is a few pixels smaller than the input
			img.shape[1] - kernel_size[1] + 1,
			kernel_size[0],
			kernel_size[1],
		),
		strides=(
			img.strides[0],
			img.strides[1],
			img.strides[0],  # When we move one step in the 3rd dimension, we should move one step in the original data too
			img.strides[1],
		),
		writeable=False,  # totally use this to avoid writing to memory in weird places
	)

def load_image(image_path):
	image = cv2.imread(image_path)
	image = image[...,::-1]
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
