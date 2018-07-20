<h1 align="center">Particle Swarm Optimizer</h1> 
<p align="center">for</p>
<h1 align="center">Neural Net Training</h1>

This is an experiment work done to remove Backpropagation and in-turn Gradient Descent 
and use Particle Swarm Optimization technique for Neural Network Training.

## What to keep in Mind?
We have all trained Neural Networks using backpropagation and we all know that it works great. 
But we all have been stuck at the point where it feels like the slow training scenario costs 
more time to go around the implementation phase of a neural network.

**By no means this approach is said to faster or more reliable than Backpropagation. This is just a fun experiment.**

## What is the Approach used?
### Particle Swarm Optimization
Particle swarm optimization is a meta-heuristics algorithm which comes 
under the sub-category of population based meta-heuristics. This means more than one 
particle is placed in the n-dimensional solution space to get to the optimum solution.

| ![PSO](/images/ParticleSwarmArrowsAnimation.gif)|
|:-----------------------------------------------:|
| *A particle swarm searching for the global minimum of a function* |
| *Source: [Wikipedia](https://en.wikipedia.org/wiki/Particle_swarm_optimization)*|

### Feedforward Neural Network
A Feedforward NN is a stack of two operations (a linear operation followed by a non-linear one) applied multiple times to find mapping from some <img src="/images/XtoY.png" height=28 width=28>.

| ![Feedforward NN operation](/images/nn_operation.png)|
| :---------------------------------------------------:|
| *Feedforward Neural Network Operation per layer* |

Here,<br/>
*Y*: is output of the neural net layer<br/>
*g(.)*: is a non-linear function<br/>
*W*: is weight matrix which basically performs a linear matrix transformation on *X*<br/>
*X*: is input to the layer<br/>
*b*: is the bias (like intercept *c* in the equation of a line).

### How to combine the two?
Suppose our NN have 3 layers with each having a weight matrix <img src="/images/layeri_w.png" height=28 width=28> and a bias vector <img src="/images/layeri_b.png" height=28 width=28> with dimension equal to the number of outputs of the layer.

*Let's flatten all the weights and biases and concatenate them into a big long vector representing a particle in the n-dimensional solution space. Where solution is the combination of weights and biases that gives the least possible error given the architecture of the network.*

| ![Background Procedure](/images/background_process.png)|
| :---------------------------------------------------:|
| *Background Procedure* |

**All the above calculations are taken care of by the modules.**

## Usage

A simple model:

```
from nn.model import Model
from nn import sigmoid, tanh
from nn.layers import Layer
from nn.losses import BinaryCrossEntropyLoss
from nn.pipeline import DataLoader

# load dataset
...

# Make a model
model = Model()
model.add_layer(Layer(2, 10, tanh)) # Layer(in_units, out_units, activation)
model.add_layer(Layer(10, 10, tanh))
model.add_layer(Layer(10, 10, tanh))
model.add_layer(Layer(10, 1, sigmoid))

# define accuracy functon
...

# compiel model
model.compile(BinaryCrossEntropyLoss, DataLoader,
    accuracy, batches_per_epoch=30, n_workers=10)

# train model
model.fit(x_train, y_train, 50)

# predict model
y_hat = model.predict(x_test)
```

Add preprocessing to mixture.

```
from nn.pipeline import DataLoader
# other imports
...

# make Data Loder class with preprocessing 
class DataLoaderWithPreprocess(DataLoader):

    def preprocess(self, X, y=None):
        if y is None:
            return X / 255.
        return X / 255., y

# make a model
...

# compile the model
model.compile(BinaryCrossEntropyLoss, DataLoaderWithPreprocess,
    accuracy, batches_per_epoch=30, n_workers=10)
```

## Demo
Try ``python demo.py`` to train a model to classify between two multivariate normal distribution.

### Experiment Failed
*This script failed the **hello world deep learning** problem - MNIST Classification*. Script Included