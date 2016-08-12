"""

Bogoto Neural Network Tutorial with backpropogation


This code does not work in python 2.7.12

"""

##import numpy as np
##
##def Sigmoid(x):
##    """ Mathematical equation of the Sigmoid function  """
##    return 1.0/ (1.0 + np.exp(-x))
##
##def Sigmoid_Prime(x):
##    """ Mathematical equation of the derivation of the Sigmoid function """
##    return Sigmoid(x) * (1.0 - Sigmoid(x))
##
##class NeuralNetwork:
##    """ Neural Network Class that uses backpropogation"""
##    def __init__(self, layers, activation = "Sigmoid"):
##        if activation == "Sigmoid":
##            self.activation = Sigmoid
##            self.activation_prime = Sigmoid_Prime
##
##        # Set weights
##        self.weights = []
##
##        # layers = [2,2,1]
##        # range of weight values (-1, 1)
##        # input and hidden layers - random ((2+1, 2+1)) : 3 x 3
##
##        for i in range(1, len(layers) - 1):
##            r = 2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1
##            self.weights.append(r)
##
##        #output layer - random((2 + 1, 1)) : 3 x 1
##        r = 2 * np.random.random( (layers[i] + 1, layers[i + 1])) - 1
##        self.weights.append(r)
##
##    def Train(self, inputs, actual_outputs, learning_rate = 0.2, epochs = 100000):
##        """ Training a neural network with data and before making its own decisions """
##        # Add column of ones to X
##        # This is to add the bias unit to the input layer
##        ones = np.atleast_2d(np.ones(inputs.shape[0]))
##        inputs = np.concatenate((ones.T, inputs), axis = 1)
##
##        for x in range(epochs):
##            if x % 10000 == 0:
##                print "Epochs: ", x
##
##                # Hidden Layer
##                i = np.random.randint(inputs.shape[0])
##                a = [inputs[i]]
##
##                for l in range(len(self.weights)):
##                    dot_value = np.dot(a[l], self.weights[l])
##                    activation = self.activation(dot_value)
##                    a.append(activation)
##
##                # Output layer
##                error = actual_outputs[i] - a[-1]
##                deltas = [error * self.activation_prime(a[-1])]
##
##                # We need to begin at the second to last layer
##                # (A layer before the output layer)
##                for l in range(len(a) - 2, 0, -1):
##                    deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_prime(a[l]))
##
##                # Reverse
##                # [level3(output) -> level2(hidden)] => [level2(hidden) -> level3(output)]
##                deltas.reverse()
##
##                # Backpropagation
##                # 1. Multiply its output delta and input activation
##                #    to get the gradient of the weight
##                # 2. Subtract a ratio (percentage) of the gradient from the weight.
##                for i in range(len(self.weights)):
##                    layer = np.atleast_2d(a[i])
##                    delta = np.atleast_2d(deltas[i])
##                    self.weights[i] += learning_rate * layer.T.dot(delta)
##
##    def Predict(self, x):
##        a = np.concatenate((np.ones(1).T, np.array(x)), axis = 1)
##        for l in range(0, len(self.weights)):
##            a = self.activation(np.dot(a, self.weights[l]))
##        return a
##
##if __name__ == "__main__":
##    """ Train XOR Gate with Neural Network """
##    nn = NeuralNetwork([2,2,1])
##
##    inputs = np.array([[0, 0],
##                       [0, 1],
##                       [1, 0],
##                       [1, 1]])
##    
##    actual_outputs = np.array([0, 1, 1, 0])
##
##    nn.Train(inputs, actual_outputs)
##
##    for e in inputs:
##        print(e, nn.Predict(e))
##    
##
##

import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2


class NeuralNetwork:

    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        # Set weights
        self.weights = []
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
        # output layer - random((2+1, 1)) : 3 x 1
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)

    def fit(self, X, y, learning_rate=0.2, epochs=100000):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
         
        for k in range(epochs):
            if k % 10000 == 0: print 'epochs:', k
            
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)
            # output layer
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            # we need to begin at the second to last layer 
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiply its output delta and input activation 
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x): 
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=1)      
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

if __name__ == '__main__':

    nn = NeuralNetwork([2,2,1])

    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y = np.array([0, 1, 1, 0])

    nn.fit(X, y)

    for e in X:
        print(e,nn.predict(e))
