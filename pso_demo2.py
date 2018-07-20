import numpy as np
from particle import Particle
from pso import ParticleSwarmOptimizer
from matplotlib import pyplot as plt
    
mean_01 = np.array([1.0, 2.0])
mean_02 = np.array([-1.0, 4.0])

cov_01 = np.array([[1.0, 0.9], [0.9, 2.0]])
cov_02 = np.array([[2.0, 0.5], [0.5, 1.0]])

ds_01 = np.random.multivariate_normal(mean_01, cov_01, 250)
ds_02 = np.random.multivariate_normal(mean_02, cov_02, 250)

all_data = np.zeros((500, 3))
all_data[:250, :2] = ds_01
all_data[250:, :2] = ds_02
all_data[250:, -1] = 1

np.random.shuffle(all_data)

split = int(0.8 * all_data.shape[0])
x_train = all_data[:split, :2]
x_test = all_data[split:, :2]
y_train = all_data[:split, -1]
y_test = all_data[split:, -1]

def sigmoid(logit):
    return 1 / (1 + np.exp(-logit))

def fitness(w, X=x_train, y=y_train):
    logit  = w[0] * X[:, 0] + w[1] * X[:, 1] + w[2]
    preds = sigmoid(logit)

    return binary_cross_entropy(y, preds)

def binary_cross_entropy(y, y_hat):
    left = y * np.log(y_hat + 1e-7)
    right = (1 - y) * np.log((1 - y_hat) + 1e-7)
    return -np.mean(left + right)

pso = ParticleSwarmOptimizer(Particle, 0.1, 0.3, 30, fitness, 
                             lambda x, y: x<y, n_iter=100,
                             dims=3, random=True,
                             position_range=(0, 1), velocity_range=(0, 1))
pso.optimize()

print(pso.gbest, fitness(pso.gbest, x_test, y_test))