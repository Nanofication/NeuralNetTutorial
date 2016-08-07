"""

Programmer: Nannan Yao

Perceptron, simplest form of Neural Network.

Testing online tutorials and understanding what it really is good for

TUTORIAL:

Make an OR Function

 A | B | A OR B |
-----------------
 0 | 0 | 0
 0 | 1 | 1
 1 | 0 | 1
 1 | 1 | 1

"""

from random import choice
from numpy import array, dot, random
#from pylab import *

# STEP Function

unit_step = lambda x: 0 if x < 0 else 1

# Map the possible input to the expected output
# First 2 inputs are the 2 input values.
# 2nd element is the expected result
# 3rd input is a "dummy" input (BIAS)


training_data = [
    (array([0,0,1]), 0),
    (array([0,1,1]), 1),
    (array([1,0,1]), 1),
    (array([1,1,1]), 1),
]

# Random numbers between 0 and 1 as initial weights
weight = random.rand(3)

# Errors list is used to store the values to be plotted
# learningRate variable controls learning rate
# iterations is the number of iterations

errors = []
learningRate = 0.2
iterations = 500

#Target: Reduce error magnitude to zero

for i in xrange(iterations):
    x, expected = choice(training_data)
    result = dot(weight,x)
    error = expected - unit_step(result)
    errors.append(error)
    weight += learningRate * error * x

for x, _ in training_data:
    result = dot(x, weight)
    print("{}: {} -> {}".format(x[:2], result, unit_step(result)))

#PLOT Errors

#ylim([-1,1])
#plot(errors)




