"""
Tutorial neural network done by Ryan Harris


"""

#
# Imports
#

import numpy as np

class BackPropogationNetwork:
    """ A back-propogation network """

    #
    # Class members
    #

    layerCount = 0
    shape = None
    
    # Weights of the neural network is an empty list
    weights = []

    #
    # Class methods
    #

    def __init__(self, layerSize):
        """ Initialize the network """

        # Layer Info
        self.layerCount = len(layerSize) - 1
        # Input layer is a buffer that holds information 1 less that length of the tuple passed in
        self.shape = layerSize
        # Tuple that the network will spit out the shape of the network

        # Input/ Output data from the last Run
        self.layerInput = []
        self.layerOutput = []

        """
        Note to self:

        list(zip(A[:-1],A[1:])) = [(2,3),(3,2)]
        each set is zipped together
        each of these weights, we create a matrix.
        
        """

        # Create the weight arrays
        for (layerOne, layerTwo) in zip(layerSize[:-1], layerSize[1:]):
            self.weights.append(np.random.normal(scale = 0.1, size = (layerTwo, layerOne + 1)))

#       # Do matrix multiplication with the 3 columns. +1 for the bias node


#
# If run as a script, create a test object
#

if __name__ == "__main__":
    # 2 inputs 2 hidden layers 1 output layer
    bpn = BackPropogationNetwork((2,2,1))
    print bpn.shape
    print bpn.weights
