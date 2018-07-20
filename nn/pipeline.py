import numpy as np
from copy import deepcopy

class DataLoader:

    def __init__(self, X, y=None, batch_size=32, repeat=True, shuffle=True):
        self.X = deepcopy(X)
        self.y = deepcopy(y)

        if self.y is not None and len(self.y.shape) == 1:
            self.y = np.expand_dims(self.y, axis=1)

        self.batch_size = batch_size
        self.repeat = repeat
        self.shuffle = shuffle

    def _shuffle(self):
        if self.y is not None:
            indices = np.arange(0, self.X.shape[0])
            np.random.shuffle(indices)
            self.X = self.X[indices]
            self.y = self.y[indices]
        else:
            np.random.shuffle(self.X)

    def preprocess(self, X, y=None):
        if y is None:
            return X
        return X, y

    def _inf_generator(self):
        m = self.X.shape[0]
        while True:
            if self.shuffle:
                self._shuffle()

            for i in range(0, m, self.batch_size):
                if i + self.batch_size > m:
                    t_index = i
                    b_index = self.batch_size-(m-i)
                    X = np.vstack((self.X[t_index:], self.X[:b_index]))
                    
                    if self.y is not None:
                        y = np.vstack((self.y[t_index:], self.y[:b_index]))
                        yield self.preprocess(X, y)
                    else:
                        yield self.preprocess(X)
                else:
                    X = self.X[i:i+self.batch_size]

                    if self.y is not None:
                        yield self.preprocess(X, self.y[i:i+self.batch_size])
                    else:
                        yield self.preprocess(X)

    def _one_shot_generator(self):
        m = self.X.shape[0]
        if self.shuffle:
            self._shuffle()
        for i in range(0, m, self.batch_size):
            if i + self.batch_size > m:
                X = self.X[i:]
                
                if self.y is not None:
                    yield self.preprocess(X, self.y[i:])
                else:
                    yield self.preprocess(X)
            else:
                X = self.X[i:i+self.batch_size]

                if self.y is not None:
                    yield self.preprocess(X, self.y[i:i+self.batch_size])
                else:
                    yield self.preprocess(X)

    def get_generator(self):
        if self.repeat:
            return self._inf_generator()
        return self._one_shot_generator()
