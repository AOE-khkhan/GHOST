# from thrid party lib
import cv2
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
	return image