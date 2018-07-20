from nn import softmax
from nn.model import Model
from nn.layers import Layer
from nn.losses import CrossEntropyLoss
from nn.pipeline import DataLoader
import numpy as np
import pandas as pd

def one_hot(y, depth=10):
    y_1hot = np.zeros((y.shape[0], 10))
    y_1hot[np.arange(y.shape[0]), y] = 1
    return y_1hot

df = pd.read_csv('./datasets/train.csv')
all_data = df.values

np.random.shuffle(all_data)

split = int(0.8 * all_data.shape[0])
x_train = all_data[:split, 1:]
x_test = all_data[split:, 1:]
y_train = all_data[:split, 0]
y_test = all_data[split:, 0]

y_train = one_hot(y_train.astype('int'))
y_test = one_hot(y_test.astype('int'))

def accuracy(y, y_hat):
    y = np.argmax(y, axis=1)
    y_hat = np.argmax(y_hat, axis=1)

    return np.mean(y==y_hat)

def relu(x):
    return np.maximum(x, 0)

model = Model()
model.add_layer(Layer(784, 10, softmax))
#model.add_layer(Layer(64, 64, relu))
#model.add_layer(Layer(64, 10, softmax))

model.compile(CrossEntropyLoss, DataLoader, accuracy,
              batches_per_epoch=x_train.shape[0] // 32 + 1,
              n_workers=50, c1=1., c2=2.)
model.fit(x_train, y_train, 100)
y_hat = model.predict(x_test)

print('Accuracy on test:', accuracy(y_test, y_hat))