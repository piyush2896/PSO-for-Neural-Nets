import numpy as np
from particle import Particle
from pso import ParticleSwarmOptimizer

def fn(x):
    return 5 * x + 3

def mse(x):
    x0 = np.random.uniform(-100, 100, 1000)
    y = fn(x0)
    y_hat = x[0] * x0 + x[1]
    return np.mean((y - y_hat) ** 2)

pso = ParticleSwarmOptimizer(Particle, 2., 2., 10, mse, 
                             lambda x, y: x<y, n_iter=100,
                             dims=2, random=True,
                             position_range=(0, 10), velocity_range=(0, 1))
pso.optimize()
print(pso.gbest)