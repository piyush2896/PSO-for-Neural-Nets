from collections import Iterable
import numpy as np

class Particle:

    def __init__(self, random, position=[0.],
                 velocity=[0.], position_range=None,
                 velocity_range=None, dims=None, alpha=0.1):
        self._validate(random, position, velocity, position_range, velocity_range, dims, alpha)

        self.random = random
        self.position = position
        self.velocity = velocity
        self.position_range = position_range
        self.velocity_range = velocity_range
        self.dims = dims
        self.alpha=alpha

        self._init_particle()

        self.pbest = self.position

    def _validate(self, random, position,
                 velocity, position_range,
                 velocity_range, dims, alpha):
        if not isinstance(random, bool):
            raise TypeError('random should be of type bool but got type {}'.format(type(random)))
        if not isinstance(alpha, float):
            raise TypeError('alpha should be of type float but got type {}'.format(type(alpha)))

        if random is True:
            if not isinstance(position_range, Iterable):
                raise TypeError('When random is True position_range should be an'
                                 ' Iterable of length 2 but got {}.'.format(type(position_range)))
            if not isinstance(velocity_range, Iterable):
                raise TypeError('When random is True velocity_range should be an'
                                 ' Iterable of length 2 but got {}.'.format(type(position_range)))
            if not isinstance(dims, int):
                raise TypeError('When random is True dims should be an'
                                 ' int but got {}.'.format(type(position_range)))
        elif random is False:
            if not isinstance(position, Iterable):
                raise TypeError('When random is False position should be an'
                                 ' Iterable but got {}.'.format(type(position_range)))
            if not isinstance(velocity, Iterable):
                raise TypeError('When random is False velocity should be an'
                                 ' Iterable but got {}.'.format(type(position_range)))

    def _init_particle(self):
        if self.random:
            self.position = np.random.uniform(low=self.position_range[0],
                                              high=self.position_range[1],
                                              size=(self.dims,))
            self.velocity = np.random.uniform(low=-abs(self.velocity_range[1]-self.velocity_range[0]),
                                              high=abs(self.velocity_range[1]-self.velocity_range[0]),
                                              size=(self.dims,))
        else:
            self.position = np.asarray(position)
            self.velocity = np.asarray(velocity)
            self.dims = self.position.shape[0]

    def update(self, c1, c2, gbest, fitness_fn, compare_fn):
        if not isinstance(c1, float):
            raise TypeError('c1 should be of type float but got {}'.format(type(c1)))
        if not isinstance(c2, float):
            raise TypeError('c2 should be of type float but got {}'.format(type(c2)))
        if not isinstance(gbest, type(self.position)):
            raise TypeError('gbest should have same type as Particle\'s velocity,'
                            'which is of type {}'.format(type(self.velocity)))
        if self.position.shape[0] != gbest.shape[0]:
            raise ValueError('gbest should have shape {} but got shape {}'.format(
                self.position.shape, gbest.shape))

        self._update_velocity(c1, c2, gbest)
        self._update_position(fitness_fn, compare_fn)

    def _update_velocity(self, c1, c2, gbest):
        self.alpha = self.alpha / 2
        wrt_pbest = c1 * np.random.rand() * (self.pbest - self.position)
        wrt_gbest = c2 * np.random.rand() * (gbest - self.position)
        self.velocity = self.alpha * self.velocity + wrt_pbest + wrt_gbest

    def _update_position(self, fitness_fn, compare_fn):
        self.position = self.position + self.velocity + 0.01 * self.position
        if compare_fn(fitness_fn(self.position), fitness_fn(self.pbest)):
            self.pbest = self.position

    def __repr__(self):
        return '<Particle: dims={} random={}>'.format(self.dims, self.random)
