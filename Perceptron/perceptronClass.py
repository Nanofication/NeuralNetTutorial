"""

2nd Perceptron Tutorial

Perceptron class

Inputs * Weights -> Net Input Function -> Activtion Function -> Output

"""

import numpy as np

class Perceptron(object):

    def __init__(self, learningRate = 0.01, iterations = 50):
        self.learningRate = learningRate
        self.iterations = iterations

    def Train(self, X, y):

        self.weight = np.zeros(1 + X.shape[1])
        self.errors = []

        for i in range(self.iterations):
            errors = 0
            for x, target in zip(X,y):
                update = self.learningRate * (target - self.predict(x))
                self.weight[1:] += update * x
                self.weight[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def NetInput(self, X):
        return np.dot(X, self.weight[1:]) + self.weight[0]

    def predict(self, X):
        return np.where(self.NetInput(X) >= 0.0, 1, -1)

    
