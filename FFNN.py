import random
from AutoDiff import Value
import numpy as np
from sklearn.datasets import make_moons, make_blobs

import math

class ActivationFunctions:
    def linear(x):
        return x
    
    def relu(x):
        return max(0, x)
    
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
    def tanh(x):
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
    
    def softmax(x):
        exp_x = [math.exp(xi) for xi in x]
        sum_exp_x = sum(exp_x)
        return [expi / sum_exp_x for expi in exp_x]

class LossFunctions:
    def mse(y_true, y_pred):
        n = len(y_true)
        squared_error = sum([(y_true[i] - y_pred[i]) ** 2 for i in range(n)])
        return squared_error / n
    
    def binary_cross_entropy(y_true, y_pred):
        n = len(y_true)
        loss = 0
        for i in range(n):
            loss += (y_true[i] * math.log(y_pred[i]) + (1 - y_true[i]) * math.log(1 - y_pred[i]))
        return -loss / n
    
    def categorical_cross_entropy(y_true, y_pred):
        n = len(y_true)
        c = len(y_true[0]) 
        loss = 0
        for i in range(n):
            for j in range(c):
                loss += y_true[i][j] * math.log(y_pred[i][j])
        return -loss / n

class Neuron():
    def __init__(self, n_input, activation_function="linear"):
        self.weight = [Value(random.uniform(-1,1)) for _ in range(n_input)]
        self.bias = Value(0)
        self.activation_function = activation_function

    def __call__(self, x):
        if isinstance(x[0], list):  # Batch input
            return [self(single_x) for single_x in x]
        
        c = sum((wi * xi for wi, xi in zip(self.weight, x)), self.bias)
        
        if (self.activation_function=="linear"):
            return c
        elif (self.activation_function=="relu"):
            return c.relu()
        elif (self.activation_function=="sigmoid"):
            return 1/(1 + (-c).exp())
        else:
            return c # Default

    def parameters(self):
        return self.weight + [self.bias]
    
    def reset_gradient(self):
        for p in self.parameters():
            p.gradient = 0

class Layer():
    def __init__(self, n_input, n_output, **kwargs):
        self.neurons = [Neuron(n_input, **kwargs) for _ in range(n_output)]

    def __call__(self, x):
        if isinstance(x[0], list):  # Batch input
            return [[n(single_x) for n in self.neurons] for single_x in x]
        
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def reset_gradient(self):
        for p in self.parameters():
            p.gradient = 0

class FFNN():
    def __init__(self, layers, activation_functions, loss_function, weight_initializer, batch_size, learning_rate, epoch, verbose):
        self.layers = [Layer(layers[i], layers[i+1], activation_function=activation_functions[i]) for i in range(len(layers)-1)]
        self.loss_function = loss_function
        self.weight_initializer = weight_initializer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.verbose = verbose

    def __call__(self, x):
        if isinstance(x[0], list):
            for layer in self.layers:
                x = [layer(sample) for sample in x]
            return x
        
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def reset_gradient(self):
        for p in self.parameters():
            p.gradient = 0
    
    def visualize_graph(self):
        return
    
    def visualize_weight_distribution(self, layers):
        return
    
    def visualize_weight_gradient_distibution(self, layers):
        return
    
    def fit(self, X, y):
        if self.batch_size is None:
            Xb, yb = X, y
        else:
            ri = np.random.permutation(X.shape[0])[:batch_size]
            Xb, yb = X[ri], y[ri]
        inputs = [list(map(Value, xrow)) for xrow in Xb]
        
        for k in range(self.epoch):
            outputs = list(map(self, inputs))
            losses = [(y - output)**2 for y, output in zip(yb, outputs)]
            loss = sum(losses)/len(losses)
            self.reset_gradient()
            loss.updateGradients() # Back propagation
            for p in self.parameters():
                p.value -= p.gradient * self.learning_rate
                
            if (self.verbose==1):
                print(f'Epoch {k}, Loss: {loss.value}')
        return

    def predict(self, X):
        inputs = [list(map(Value, xrow)) for xrow in X]
        outputs = [1 if self(x).value > 0 else 0 for x in inputs]
        return outputs
    
    def save():
        return
    
    def load():
        return

# Test
# if __name__ == "__main__":
#     X, y = make_moons(n_samples=100, noise=0.1)
    
#     model = FFNN(layers=[16, 16, 1], activation_functions=["relu", "linear"], loss_function="mse", weight_initializer="random", batch_size=None, learning_rate=0.1, epoch=1000, verbose=1)
#     model.fit(X, y)
#     print(model.predict(X))
#     accuracy = sum(model.predict(X) == y) / len(y)
#     print(accuracy)