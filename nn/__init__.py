import numpy as np

def relu(x):
    return np.maximum(x, 0)

def leaky_rely(x, alpha=0.01):
    nonlin = relu(x)
    nonlin[nonlin==0] = alpha * x[nonlin == 0]
    return nonlin

def sigmoid(x):
    return 1 / (1 + np.exp(x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def tanh(x):
    return np.tanh(x)