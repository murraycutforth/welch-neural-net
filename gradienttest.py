import numpy as np

def computeNumericalGradient(NN, X, y):
    # This function applies the central difference approximation to estimate the gradient
    # of the cost function wrt the weights of the neural network.
    paramsInitial = NN.getParams()
    grad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    eps = 1e-4

    for p in range(len(paramsInitial)):
        # Perturb in a single dimension at a time
        perturb[:] = 0.0
        perturb[p] = eps
        NN.setParams(paramsInitial + perturb)
        costplus = NN.costFunction(X,y)

        NN.setParams(paramsInitial - perturb)
        costminus = NN.costFunction(X,y)

        grad[p] = (costplus - costminus) / (2 * eps)

    NN.setParams(paramsInitial)
    return grad
