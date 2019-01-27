import numpy as np

# https://enlight.nyc/projects/neural-network/: the url of the boilerplate code for neural network

N = 3

# data
feature_set = np.array(list(([x] for x in range(2, N))), dtype=float)
labels = np.array([[x+2 for x in range(2, N)]])  
xPredicted = np.array(([N+1]), dtype=float)

# for x in [feature_set, labels, xPredicted]:
#   print(x)

# feature_set = np.array(([5, 1, 1], [1, 5, 1], [1, 1 ,5]), dtype=float)
# labels = np.array(([1, 5, 1], [1, 1, 5], [1, 1, 5]), dtype=float)
# xPredicted = np.array(([5]), dtype=float)

# feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])  
# labels = np.array([[1,0,0,1,1]])  
# xPredicted = np.array(([0,1,0], [1,0,0]), dtype=float)


# np.random.seed(42)  
weights = np.random.rand(1,1)  
bias = np.random.randint(-10, 10)  
lr = 0.1  

print(weights, bias)

def sigmoid(x):
  a = x.copy()
  a[a < 0] = 0
  return a
  # return 1/(1+np.exp(-x))

def sigmoid_der(x):  
  a = x.copy()
  a[a < 0] = 0
  return a
  # return sigmoid(x)*(1-sigmoid(x))
  # return sigmoid(x)*(1-sigmoid(x))

le = 0
while True:  
  inputs = feature_set

  # feedforward step1
  XW = np.dot(feature_set, weights) + bias

  #feedforward step2
  z = sigmoid(XW)


  # backpropagation step 1
  error = z - labels
  e = error.sum()
  print('output = {}, error = {}\n'.format(z, e))

  if abs(e - le) < 0.0001:
    break
  le = e

  # backpropagation step 2
  dcost_dpred = error
  dpred_dz = sigmoid_der(z)

  z_delta = dcost_dpred * dpred_dz

  inputs = feature_set.T
  weights -= lr * np.dot(inputs, z_delta)

  for num in z_delta:
    bias -= lr * num

for single_point in xPredicted:
  result = sigmoid(np.dot(single_point, weights) + bias)  
  print(result) 

