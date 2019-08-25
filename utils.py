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
	# image = image[...,::-1]
	image = np.array(image, dtype='float64')
	return image

def resultant(matrix):
	return round(np.sum(matrix), 4) 

def get_similarity_ratio(a, b):
	return (255**-1) - abs(a - b).mean()

def toGrey(img, r=1/3, g=1/3, b=1/3):
# def toGrey(img, r=0.299, g=0.587, b=0.114):
	s = img.shape
	if len(s) == 2 or (len(s) == 3 and s[-1] < 3):
		return img
	return np.add(b*img[:, :, 0], g*img[:, :, 1], r*img[:, :, 2])

def is_row_in_array(row , arr):
	return np.where((row == arr).all(tuple(range(len(arr.shape)))[1:]) == True)[0]

def validateFolderPath(folder_path):
	if not os.path.exists(folder_path):
		os.mkdir(folder_path)