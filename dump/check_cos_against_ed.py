import numpy as np

def main():
	n = 10
	for ni in range(n):
		size = np.random.randint(3, 4, 1)[0]
		sample_image1 = np.random.randint(0, 5, size).flatten()
		sample_image2 = np.random.randint(0, 5, size).flatten()

		print(sample_image1)
		print(sample_image2)

		cs = cosine_similarity(sample_image1, sample_image2)
		edr = euclidean_distance_ratio(sample_image1, sample_image2)

		print(f'iteration {ni} of matrix of size {size}: cosine sim = {cs}, euclidean dist ratio = {edr}')
		print()

def cosine_similarity(image1, image2):
	return np.dot(image1, image2) / (euclidean_distance(image1) * euclidean_distance(image2))

def euclidean_distance(image):
	return np.sqrt(np.sum(image**2))

def euclidean_distance_ratio(image1, image2, max_value=255):
	max_matrix = max_value * np.ones(image1.shape[0])

	max_euclidean_distance = euclidean_distance(max_matrix)

	diff = abs(image2 - image1)
	ed_diff = euclidean_distance(diff)

	print(1 - (abs(euclidean_distance(image1) - euclidean_distance(image2)) / max_euclidean_distance))
	return 1 - (ed_diff / max_euclidean_distance)

if __name__ == '__main__':
	main()