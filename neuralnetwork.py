import numpy as np

class Neural_Network():
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

    # Functions for interacting with other methods / classes

    def getParams(self):
        # Get all weights rolled into a single vector
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        # Set W1 and W2 using single vector of parameters
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W1 = np.reshape(params[W1_start : W1_end],
                             (self.inputLayerSize, self.hiddenLayerSize))
        self.W2 = np.reshape(params[W1_end : W2_end],
                             (self.hiddenLayerSize, self.outputLayerSize))

    def getGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


class Regularised_Neural_Network(Neural_Network):

    def __init__(self, Lambda):
        super(Regularised_Neural_Network, self).__init__()
        self.Lambda = Lambda

    def costFunction(self, X, Y):
        # Compute sum-squared errors on Y
        return 0.5 * np.sum((Y - self.forward(X))**2)/X.shape[0] \
               + 0.5 * self.Lambda * (np.sum(self.W1**2) + np.sum(self.W2**2))

    def costFunctionPrime(self, X, Y):
        # Compute partial derivatives of cost function wrt W1 and W2
        yHAT = self.forward(X)

        delta3 = np.multiply(yHAT - Y, self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3) + self.Lambda * self.W2

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2) + self.Lambda * self.W1

        return dJdW1, dJdW2