import numpy as np

class Neural_Network(object):
    def __init__(self):
        # Define hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        # Initialise weights on each synapse randomly from standard normal dist
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        # Propagate given input through network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHAT = self.sigmoid(self.z3)
        return yHAT

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
