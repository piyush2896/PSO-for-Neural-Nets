import numpy as np
from swarm_intelligence.pso import *
from functools import partial

class Model:

    def __init__(self):
        self.layers = []
        self.n_wts = []
        self.compiled = False

    def add_layer(self, layer):
        self.layers.append(layer)
        self.n_wts.append(layer.n_wts)

    def _calc_dims(self):
        return int(np.sum(self.n_wts))

    def compile(self, loss_fn, dataloader_cls, metric_fn=None, c1=2.,
                c2=2., n_workers=10, batch_size=32, batches_per_epoch=100,
                position_range=(-1, 1), velocity_range=(-1, 1)):

        self.dataloader_cls = dataloader_cls
        self.data_loader = partial(dataloader_cls, batch_size=batch_size,
                                  repeat=True, shuffle=True)
        self.metric_fn = metric_fn

        self.loss_fn = partial(loss_fn, layers=self.layers, n_wts=self.n_wts, dims=self._calc_dims())

        self.optimizer = partial(ParticleSwarmOptimizer, particle_cls=Particle, c1=c1,
            c2=c2, n_particles=n_workers, compare_fn=lambda x, y: x < y,
            n_iter=batches_per_epoch, dims=self._calc_dims(), random=True,
            position_range=position_range, velocity_range=velocity_range)

        self.compiled = True

    def _forward(self, X, wts):
        w_index = 0
        for i, layer in enumerate(self.layers):
            X = layer.forward(wts[w_index:w_index+self.n_wts[i]], X)
            w_index += self.n_wts[i]
        return X

    def fit(self, X, y, epochs=1):
        assert self.compiled, 'Call compile before training'

        data_loader = self.data_loader(X=X, y=y).get_generator()
        loss_fn = self.loss_fn(data_loader=data_loader)
        if isinstance(self.optimizer, partial):
            self.optimizer = self.optimizer(fitness_fn=loss_fn)

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch+1, epochs))
            self.optimizer.optimize()
            if self.metric_fn is not None:
                print('Metric: {}'.format(self.metric_fn(y, self.predict(X))), end='\t')
            print('Loss:', self._loss(X, y))

    def predict(self, X):
        assert self.compiled, 'Call compile before Prediction'

        data_loader = self.dataloader_cls(X,
            batch_size=32, repeat=False, shuffle=False).get_generator()
        y = []
        for X in data_loader:
            y.append(self._forward(X, self.optimizer.gbest))
        return np.vstack(y)

    def _loss(self, X, y):
        data_loader = self.dataloader_cls(X, y,
            batch_size=32, repeat=False, shuffle=False).get_generator()
        loss_fn = self.loss_fn(data_loader=data_loader)
        y = []
        try:
            while True:
                y.append(loss_fn(self.optimizer.gbest))
        except StopIteration:
            return np.mean(y)