import numpy as np
from sklearn.datasets import make_moons, make_blobs
import pickle
import math
from collections import defaultdict

class Neuron():
    def __init__(self, n_input, activation_function="linear"):
        self.activation_function = activation_function
        self.n_input = n_input
        self.weight = np.zeros(n_input)
        self.bias = 0
        self.output = 0
        self.input = None
        self.gradient_weight = np.zeros(n_input)
        self.gradient_bias = 0

    def __call__(self, x):
        self.input = np.array(x)
        z = np.dot(self.weight, self.input) + self.bias
        
        if self.activation_function == "linear":
            self.output = z
        elif self.activation_function == "relu":
            self.output = np.maximum(0, z)
        elif self.activation_function == "sigmoid":
            self.output = 1 / (1 + np.exp(-z))
        elif self.activation_function == "tanh":
            self.output = np.tanh(z)
        elif self.activation_function == "softmax":
            exp_z = np.exp(z - np.max(z))
            self.output = exp_z / exp_z.sum()
        return self.output

    def backward(self, gradient_output):
        if self.activation_function == "relu":
            gradient_activation = gradient_output * (self.output > 0)
        elif self.activation_function == "sigmoid":
            gradient_activation = gradient_output * (self.output * (1 - self.output))
        elif self.activation_function == "tanh":
            gradient_activation = gradient_output * (1 - self.output ** 2)
        # elif self.activation_function == "softmax":
        else:
            gradient_activation = gradient_output
        
        self.gradient_weight += gradient_activation * self.input
        self.gradient_bias += gradient_activation

    def update(self, learning_rate, batch_size):
        self.weight -= learning_rate * self.gradient_weight / batch_size
        self.bias -= learning_rate * self.gradient_bias / batch_size
        self.gradient_weight.fill(0)
        self.gradient_bias = 0

class Layer():
    def __init__(self, n_input, n_output, activation_function="linear"):
        self.neurons = [Neuron(n_input, activation_function) for _ in range(n_output)]
    
    def __call__(self, x):
        return np.array([n(x) for n in self.neurons])
    
    def backward(self, gradient_output):
        gradient_input = np.zeros(len(self.neurons[0].weight))
        for i, neuron in enumerate(self.neurons):
            neuron.backward(gradient_output[i])
            gradient_input += neuron.weight * gradient_output[i]
        return gradient_input
    
    def update(self, learning_rate, batch_size):
        for neuron in self.neurons:
            neuron.update(learning_rate, batch_size)

class FFNN():
    def __init__(self, layers, activation_functions, loss_function="mse", learning_rate=0.1, epoch=10, batch_size=10, verbose=1, random_state=0):
        self.layers = [Layer(layers[i], layers[i+1], activation_functions[i]) for i in range(len(layers)-1)]
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.verbose = verbose
        self.rng = np.random.default_rng(random_state)
        self.label_to_index = {}
        self.index_to_label = {}
        
    def weight_initializer(self, weight_initializer="zero", seed=0, lower_bound=-1, upper_bound=1, mean=0, variance=0.1):
        rng = np.random.default_rng(seed)
        for layer in self.layers:
            for neuron in layer.neurons:
                if weight_initializer == "zero":
                    weights = [0 for _ in range(neuron.n_input+1)]
                elif weight_initializer == "uniform":
                    weights = rng.uniform(lower_bound, upper_bound, size=neuron.n_input+1)
                elif weight_initializer == "normal":
                    weights = rng.normal(mean, variance, size=neuron.n_input+1)
                else:  # Default = zero
                    weights = [0 for _ in range(neuron.n_input+1)]
                
                neuron.weight = np.array(weights[:-1])
                neuron.bias = weights[-1]
    
    def encode_labels(self, y):
        unique_labels = np.unique(y)
        self.label_to_index = {label: i for i, label in enumerate(unique_labels)}
        self.index_to_label = {i: label for label, i in self.label_to_index.items()}
        return np.array([self.label_to_index[label] for label in y])
    
    def decode_labels(self, y_pred):
        return np.array([self.index_to_label[int(i)] for i in np.ravel(y_pred)])

    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def compute_loss(self, y_true, y_pred):
        if self.loss_function == "mse":
            return np.mean((y_true - y_pred) ** 2)
        elif self.loss_function == "binary_cross_entropy":
            return -np.mean(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))
        elif self.loss_function == "categorical_cross_entropy":
            return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))
    
    def compute_loss_gradient(self, y_true, y_pred):
        if self.loss_function == "mse":
            return -2 * (y_true - y_pred) / len(y_true)
        elif self.loss_function == "binary_cross_entropy":
            return (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-9)
        elif self.loss_function == "categorical_cross_entropy":
            return y_pred - y_true
    
    def fit(self, X, y):
        y = self.encode_labels(y)
        n_samples = len(X)
        for epoch in range(self.epoch):
            total_loss = 0
            indices = self.rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                outputs = np.array([self(x) for x in X_batch])
                loss = self.compute_loss(y_batch, outputs)
                total_loss += loss
                
                gradients = self.compute_loss_gradient(y_batch, outputs)
                for j, x in enumerate(X_batch):
                    gradient = gradients[j]
                    for layer in reversed(self.layers):
                        gradient = layer.backward(gradient)
                
                for layer in self.layers:
                    layer.update(self.learning_rate, self.batch_size)
            if self.verbose == 1:
                print(f"Epoch {epoch+1}, Loss: {total_loss / (n_samples / self.batch_size)}")
    
    def predict_proba(self, X):
        return np.array([self(x) for x in X])
    
    def predict(self, X):
        probas = self.predict_proba(X)
        if self.loss_function == "categorical_cross_entropy":
            return self.decode_labels(np.argmax(probas, axis=1))
        return self.decode_labels((probas >= 0.5).astype(int))
