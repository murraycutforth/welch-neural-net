"""
This code follows the intro to neural networks video series at https://www.youtube.com/watch?v=bxe2T-V8XRs
"""

import numpy as np
import neuralnetwork

X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
Y = np.array(([75], [82], [93]), dtype=float)

"""
The goal is to create a neural network which can predict the test score given
a new piece of data describing the hours of sleep and hours of study for that test.
This is an example of a supervised regression problem, since the output
is continuous and we have training data with inputs (hours of sleep, hours of study) 
and outputs (test score).
"""

X = X / np.amax(X, axis=0)
Y = Y / 100.0

"""
The input and output data should be normalised. This is helpful for ensuring that the
neural network converges quicker to the minimum error.
"""

NN = neuralnetwork.Neural_Network()
print("The cost function for this training set with random initial weights is: {}"
      .format(NN.costFunction(X, Y)))
dJdW1, dJdW2 = NN.costFunctionPrime(X, Y)
print("The partial derivative of cost wrt W1 is \n {}".format(dJdW1))
print("The partial derivative of cost wrt W2 is \n {}".format(dJdW2))