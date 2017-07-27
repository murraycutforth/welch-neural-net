from scipy import optimize
import numpy as np

class trainer():
    def __init__(self, NN):
        # Store local reference to Neural Network
        self.NN = NN

    def costFunctionWrapper(self, params, X, y):
        self.NN.setParams(params)
        cost = self.NN.costFunction(X, y)
        dJdW1, dJdW2 = self.NN.costFunctionPrime(X, y)
        return cost, np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

    def callBackFunction(self, params):
        # Used to store the new cost after each iteration
        self.NN.setParams(params)
        self.J.append(self.NN.costFunction(self.X, self.y))

    def train(self, X, y):
        # Internal variable for callback function
        self.X = X
        self.y = y

        # Store costs in this list
        self.J = []

        params0 = self.NN.getParams()
        options = {'maxiter' : 200, 'disp' : True}

        _res = optimize.minimize(self.costFunctionWrapper, params0,
                                 args=(X,y), method='BFGS', jac=True,
                                 callback=self.callBackFunction, options=options)

        self.NN.setParams(_res.x)
        self.minimisationresults = _res