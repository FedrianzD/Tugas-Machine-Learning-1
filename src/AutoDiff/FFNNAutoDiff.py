from AutoDiff import Value
import numpy as np
from sklearn.datasets import make_moons, make_blobs
import pickle
import math
from graphviz import Digraph
import matplotlib.pyplot as plt
import seaborn as sns

class Neuron():
    def __init__(self, n_input, activation_function="linear"):
        self.activation_function = activation_function
        self.n_input = n_input
        self.weight = [Value(0) for i in range(n_input)]
        self.bias = Value(0)
        self.c = 0

    def __call__(self, x):
        c = sum((wi * xi for wi, xi in zip(self.weight, x)), self.bias)
        
        # Activation functions
        if (self.activation_function=="linear"):
            self.c = c
        elif (self.activation_function=="relu"):
            self.c = c.relu()
        elif (self.activation_function=="sigmoid"):
            self.c = 1/(1 + (-c).exp())
        elif (self.activation_function=="tanh"):
            self.c = (c.exp() - (-c).exp()) / (c.exp() + (-c).exp())
        elif (self.activation_function=="softmax"):
            exp_c = [ci.exp() for ci in c]
            sum_exp_c = sum(exp_c)
            self.c = [expi / sum_exp_c for expi in exp_c]
        else:
            self.c = c.relu() # Default = relu
        
        return self.c

    def parameters(self):
        return [self.bias] + self.weight
    
    def reset_gradient(self):
        for p in self.parameters():
            p.gradient = 0

class Layer():
    def __init__(self, n_input, n_output, activation_function="linear"):
        self.neurons = [Neuron(n_input, activation_function) for _ in range(n_output)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters_matrix(self):
        return [neuron.parameters() for neuron in self.neurons]
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
    def reset_gradient(self):
        for p in self.parameters():
            p.gradient = 0

class FFNNAutoDiff():
    def __init__(self, layers=[], activation_functions="relu", loss_function=None, 
                 batch_size=None, learning_rate=0.1, epoch=10, verbose=1, random_state=0):
        self.label_encoder = None
        self.label_decoder = None
        
        self.layers = [Layer(layers[i], layers[i+1], activation_function=activation_functions[i]) for i in range(len(layers)-1)]
                
        self.n_classes = layers[-1]
        if loss_function is None:
            self.loss_function = "binary_cross_entropy" if self.n_classes == 1 else "categorical_cross_entropy"
        else:
            self.loss_function = loss_function
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.verbose = verbose
        self.rng = np.random.default_rng(random_state)

    def encode_labels(self, y):
        if np.issubdtype(np.array(y).dtype, np.integer):
            unique_labels = np.unique(y)
            self.label_encoder = {label: i for i, label in enumerate(unique_labels)}
            self.label_decoder = {i: label for label, i in self.label_encoder.items()}
            return np.array([self.label_encoder[label] for label in y])
        
        # Use sklearn's LabelEncoder if not already integers
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        encoded_labels = le.fit_transform(y)
        
        # Store encoder and decoder
        self.label_encoder = {label: code for label, code in zip(le.classes_, range(len(le.classes_)))}
        self.label_decoder = {code: label for label, code in self.label_encoder.items()}
        
        return encoded_labels

    def decode_labels(self, encoded_labels):
        if self.label_decoder is None:
            return encoded_labels
        
        return np.array([self.label_decoder.get(label, label) for label in encoded_labels])

    def weight_initializer(self, weight_initializer="zero", seed=0, lower_bound=-1, upper_bound=1, mean=0, variance=0.1):
        rng = np.random.default_rng(seed)
        for layer in self.layers:
            for neuron in layer.neurons:
                if (weight_initializer=="zero"):
                    weights = [0 for i in range(neuron.n_input+1)]
                elif (weight_initializer=="uniform"):
                    weights = rng.uniform(lower_bound, upper_bound, size=neuron.n_input+1)
                elif (weight_initializer=="normal"):
                    weights = rng.normal(mean, variance, size=neuron.n_input+1)
                else: # Default = zero
                    weights = [0 for i in range(neuron.n_input+1)]
                
                neuron.weight = [Value(w) for w in weights[:-1]]
                neuron.bias = Value(weights[-1])
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters_matrix(self):
        return [layer.parameters_matrix() for layer in self.layers]
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def reset_gradient(self):
        for p in self.parameters():
            p.gradient = 0
    
    def fit(self, X, y):
        y_encoded = self.encode_labels(y)
        
        Xb, yb = X, y_encoded
        inputs = [list(map(Value, xrow)) for xrow in Xb]
        n_samples = len(yb)
        
        # Prepare labels based on classification type
        if self.n_classes == 1:
            # Binary classification
            y_processed = yb
        else:
            # Multiclass classification
            y_processed = np.zeros((n_samples, self.n_classes))
            for i, label in enumerate(yb):
                y_processed[i, label] = 1
        
        if self.batch_size is None:
            self.batch_size = n_samples
        
        for k in range(self.epoch):
            indices = self.rng.permutation(n_samples)
            X_shuffled = [inputs[i] for i in indices]
            y_shuffled = y_processed[indices] if self.n_classes > 1 else [y_processed[i] for i in indices]
            
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                outputs = list(map(self, X_batch))
                
                if self.loss_function == "mse":
                    loss = sum([(y - output)**2 for y, output in zip(y_batch, outputs)]) / len(y_batch)
                
                # Binary Cross Entropy
                elif self.loss_function == "binary_cross_entropy":
                    loss = -1 * sum([y * output.log() + (1 - y) * (1 - output).log() for y, output in zip(y_batch, outputs)]) / len(y_batch)
                
                # Categorical Cross Entropy
                elif self.loss_function == "categorical_cross_entropy":
                    loss = -1 * sum([sum([yj * outputj.log() for yj, outputj in zip(y, output)]) 
                                     for y, output in zip(y_batch, outputs)]) / len(y_batch)
                
                else:  # Default = mse
                    loss = sum([(y - output)**2 for y, output in zip(y_batch, outputs)]) / len(y_batch)
                
                self.reset_gradient()
                loss.updateGradients()  # Back propagation
                for p in self.parameters():
                    p.value -= p.gradient * self.learning_rate
            
            if self.verbose == 1:
                print(f'Epoch {k}, Loss: {loss.value}')
        return

    def predict(self, X):
        inputs = [list(map(Value, xrow)) for xrow in X]
        
        # Binary classification
        if self.n_classes == 1:
            outputs = [1 if self(x).value > 0 else 0 for x in inputs]
        
        # Multiclass classification
        else:
            outputs = [np.argmax([out.value for out in self(x)]) for x in inputs]
        
        # Decode labels back to original format
        return self.decode_labels(outputs)
    
    def predict_proba(self, X):
        # Return probability distribution for each input
        inputs = [list(map(Value, xrow)) for xrow in X]
        
        # Binary classification
        if self.n_classes == 1:
            # Sigmoid output for binary classification
            outputs = [self(x).value for x in inputs]
        
        # Multiclass classification
        else:
            outputs = [[out.value for out in self(x)] for x in inputs]
        
        return outputs
    
    def save(self, filename='model.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    def visualize_graph(self):
        w = self.parameters_matrix()
        
        dot = Digraph(graph_attr={'rankdir': "LR", 'splines': 'line', 
                                "nodesep": '1', "ranksep": '1.5'})

        input_nodes = [(f'{0}{0}', 'b0')]  # Bias node
        for i in range(len(w[0][0]) - 1):
            input_nodes.append((f'{0}{i+1}', f'X{i+1}'))

        input_nodes.sort()

        with dot.subgraph() as s:
            s.attr(rank='same')
            prev_node = None
            for node_id, label in input_nodes:
                s.node(node_id, label, shape="ellipse", fixedsize="true", width="0.8", height="0.5")
                if prev_node:
                    dot.edge(prev_node, node_id, style="invis")
                prev_node = node_id

        for i, layer in enumerate(self.layers):
            layer_nodes = []
            
            if i != len(self.layers) - 1:
                layer_nodes.append((f'{i+1}{0}', f'b{i+1}'))
            
            # Neurons
            for j, neuron in enumerate(layer.neurons):
                if i != len(self.layers) - 1:
                    layer_nodes.append((f'{i+1}{j+1}', f'h{i+1}{j+1}'))
                else:
                    layer_nodes.append((f'{i+1}{j+1}', f'o{j+1}'))

            layer_nodes.sort()

            with dot.subgraph() as s:
                s.attr(rank='same')
                prev_node = None
                for node_id, label in layer_nodes:
                    s.node(node_id, label, shape="ellipse", fixedsize="true", width="0.8", height="0.5")
                    if prev_node:
                        dot.edge(prev_node, node_id, style="invis")
                    prev_node = node_id

            for j, neuron in enumerate(layer.neurons):
                for k, weight in enumerate(w[i][j]):
                    weight_label = "{ w = %.4f | g = %.4f }" % (weight.value, weight.gradient)
                    weight_node = f'w{i+1}{j+1}{k}'

                    dot.node(weight_node, weight_label, shape='record', width="1", height="0.5")

                    dot.edge(f'{i}{k}', weight_node, tailport="e", headport="w")
                    dot.edge(weight_node, f'{i+1}{j+1}', tailport="e", headport="w")

        return dot
    
    def visualize_weight_distribution(self, layer):
        if not (0 <= layer < len(self.layers)):
            print(f"Invalid layer index: {layer}")
            return
        fig, ax = plt.subplots(figsize=(12, 4))
        weights = [p.value for neuron in self.layers[layer].neurons for p in neuron.weight]

        bins = np.linspace(min(weights), max(weights), 10)
        print(bins)
        sns.histplot(weights, bins=bins, kde=True, ax=ax)
        ax.set_title(f'Weight Distribution - Layer {layer + 1}')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')
        ax.set_xticks(np.round(bins, 2))
        plt.tight_layout()
        plt.show()
        return
    
    def visualize_weight_gradient_distibution(self, layer):
        if not (0 <= layer < len(self.layers)):
            print(f"Invalid layer index: {layer}")
            return
        fig, ax = plt.subplots(figsize=(12, 4))
        weights = [p.gradient for neuron in self.layers[layer].neurons for p in neuron.weight]
        bins = np.linspace(min(weights), max(weights), 10)
        sns.histplot(weights, bins=bins, kde=True, ax=ax)
        ax.set_title(f'Weight Gradient Distribution - Layer {layer + 1}')
        ax.set_xlabel('Weight Gradient Value')
        ax.set_ylabel('Frequency')
        ax.set_xticks(np.round(bins, 2))
        plt.tight_layout()
        plt.show()
        return