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
        # Propagate X through network
        assert X.shape[1] == self.inputLayerSize
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHAT = self.sigmoid(self.z3)
        return yHAT

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoidPrime(self, z):
        return np.exp(-z) / (1.0 + np.exp(-z))**2

    def costFunction(self, X, Y):
        # Compute sum-squared errors on Y
        return 0.5 * np.sum((Y - self.forward(X))**2)

    def costFunctionPrime(self, X, Y):
        # Compute partial derivatives of cost function wrt W1 and W2
        yHAT = self.forward(X)

        delta3 = np.multiply(yHAT - Y, self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

