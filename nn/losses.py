import numpy as np

class Loss:

    def __init__(self, data_loader, layers, n_wts, dims):
        self.data_loader = data_loader
        self.layers = layers
        self.n_wts = n_wts
        self.dims = dims

    def _forward(self, wts):
        w_index = 0
        X, y = next(self.data_loader)
        for i, layer in enumerate(self.layers):
            X = layer.forward(wts[w_index:w_index+self.n_wts[i]], X)
            w_index += self.n_wts[i]
        return y, X

    def _loss(self, y, y_hat):
        raise NotImplementedError()

    def __call__(self, wts):
        raise NotImplementedError()

class MSELoss(Loss):

    def _loss(self, y, y_hat):
        return np.mean((y - y_hat) ** 2)

    def __call__(self, wts):
        y, y_hat = self._forward(wts)
        return self._loss(y, y_hat)

class RMSELoss(Loss):

    def _loss(self, y, y_hat):
        return np.sqrt(np.mean((y - y_hat) ** 2))

    def __call__(self, wts):
        y, y_hat = self._forward(wts)
        return self._loss(y, y_hat)

class BinaryCrossEntropyLoss(Loss):

    def _loss(self, y, y_hat):
        left = y * np.log(y_hat + 1e-7)
        right = (1 - y) * np.log((1 - y_hat) + 1e-7)
        return -np.mean(left + right)

    def __call__(self, wts):
        y, y_hat = self._forward(wts)
        return self._loss(y, y_hat)

class CrossEntropyLoss(Loss):

    def _loss(self, y, y_hat):
        return -np.mean(y * np.log(y_hat + 1e-7))

    def __call__(self, wts):
        y, y_hat = self._forward(wts)
        return self._loss(y, y_hat)
