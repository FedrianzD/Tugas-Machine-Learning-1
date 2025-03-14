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