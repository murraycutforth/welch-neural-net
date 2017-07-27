"""
This code follows the intro to neural networks video series at https://www.youtube.com/watch?v=bxe2T-V8XRs
"""

import numpy as np
import neuralnetwork
import gradienttest
import trainer
from matplotlib import pyplot as plt

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
print("Constructed 3-layer, 6-node neural network with random initial synapse weights.")
print("The cost function for this training set is initially: {}."
      .format(NN.costFunction(X, Y)))
dJdW1, dJdW2 = NN.costFunctionPrime(X, Y)

"""
Here we confirm that the analytic gradient is roughly equal to 
a central difference approximation
"""
analyticgrad = NN.getGradients(X, Y)
numericgrad = gradienttest.computeNumericalGradient(NN, X, Y)
print("The L2 error norm of the difference between the numeric and analytic gradients is: {}.".format(
    np.linalg.norm(analyticgrad - numericgrad, 2)))

"""
Now we call the BFGS algorithm to train the network
"""
print("Training network...")
T = trainer.trainer(NN)
T.train(X,Y)

"""
View the cost as a function of iteration #
"""
plt.plot(T.J)
plt.grid(True)
plt.ylabel('Cost')
plt.xlabel('Number of iterations')
plt.show()
print("After training, the gradient vector is: {}."
      .format(NN.costFunctionPrime(X, Y)))

"""
Now explore the predictions of the network for various points
in the (sleep, study) space:
"""
hoursSleep = np.linspace(0, 10, 100)
hoursStudy = np.linspace(0, 5, 100)

normalisedHoursSleep = hoursSleep / 10.0
normalisedHoursStudy = hoursStudy / 5.0

xv, yv = np.meshgrid(normalisedHoursSleep, normalisedHoursStudy)

allInputs = np.zeros((xv.size, 2))
allInputs[:, 0] = xv.ravel()
allInputs[:, 1] = yv.ravel()

print("Forward propagating {} points in (sleep, study) space..."
      .format(xv.size))
allOutputs = NN.forward(allInputs)
print("Finished.")

"""
Examine results in contour plot
"""
print(hoursStudy.shape)
yy = np.dot(hoursStudy.reshape(100, 1), np.ones((1, 100)))
xx = np.dot(hoursSleep.reshape(100, 1), np.ones((1, 100))).T

plt.pcolormesh(xx, yy, 100*allOutputs.reshape(100, 100))
plt.colorbar()
levels = np.linspace(0, 100, 21)
CS = plt.contour(xx, yy, 100*allOutputs.reshape(100, 100), levels=levels, colors='black')
plt.clabel(CS, inline=True, fontsize=12, colors='black')

plt.xlabel('Hours sleep')
plt.ylabel('Hours study')
plt.show()