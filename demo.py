from nn import sigmoid
from nn.model import Model
from nn.layers import Layer
from nn.losses import BinaryCrossEntropyLoss
from nn.pipeline import DataLoader
import numpy as np
import matplotlib.pyplot as plt

mean_01 = np.array([1.0, 2.0])
mean_02 = np.array([-1.0, 4.0])

cov_01 = np.array([[1.0, 0.9], [0.9, 2.0]])
cov_02 = np.array([[2.0, 0.5], [0.5, 1.0]])

ds_01 = np.random.multivariate_normal(mean_01, cov_01, 250)
ds_02 = np.random.multivariate_normal(mean_02, cov_02, 250)

plt.scatter(ds_01[:, 0], ds_01[:, 1], color='red')
plt.scatter(ds_02[:, 0], ds_02[:, 1], color='blue')
plt.show()

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

def accuracy(y, y_hat):
    y_hat = (y_hat >= 0.5).astype('int')
    y = y.astype('int')
    return np.mean(y_hat[:, 0] == y)

model = Model()
model.add_layer(Layer(2, 10, np.tanh))
model.add_layer(Layer(10, 10, np.tanh))
model.add_layer(Layer(10, 10, np.tanh))
model.add_layer(Layer(10, 1, sigmoid))

model.compile(BinaryCrossEntropyLoss, DataLoader, accuracy, batches_per_epoch=30, n_workers=10)
model.fit(x_train, y_train, 50)
y_hat = model.predict(x_test)

print('Accuracy on test:', accuracy(y_test, y_hat))
