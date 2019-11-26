"""
Implementation of a feed-forward neural network for a simple binary classification problem.
The layers as well as nodes and learning rate per layer can be adjusten manually.
"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

import numpy as np
import matplotlib.pyplot as mp
from IPython import display
from numpy_ffnn import *

################################################################################
# %% IMPORT DATASET
################################################################################

from sklearn.datasets import make_moons
X_train, y_train = make_moons(n_samples=5000, noise=0.1, random_state=42)
X_test, y_test = make_moons(n_samples=50, noise=0.1, random_state=28)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

################################################################################
# %% INIT AND BUILD FFNN
################################################################################

ffnn = FFNN()
res_train = []
res_test = []

################################################################################
# SET PARAMETERS IF REQUIRED
################################################################################

##### NUMBER OF HIDDEN LAYERS
ffnn.hlayers = 3

##### NODES PER HIDDEN LAYERS
ffnn.nodes = {'h0':5, 'h1':3, 'h2':2}

##### LEARNING RATES PER LAYER
ffnn.learnrates = {'h0':1e-3, 'h1':1e-3, 'h2':1e-3, 'out':1e-3}

################################################################################
# %% RUN FORWARD FEED
################################################################################

for i in range(10000):

    ##### RUN A SINGLE FOWARD STEP
    y_hat = ffnn.forward(X_train,y_train,'train')

    ##### CALCULATE AND UPDATE BACKPROPAGATION, APPEND TO LOSS LIST FOR TRAINING
    res_train.append(ffnn.backpropagate(X_train, y_train, y_hat)/len(y_train))

    ##### PREDICT TEST RESULT
    y_pred = ffnn.forward(X_test, y_test, 'test')

    ##### APPEND TO LOSS LIST FOR TESTING
    res_test.append(log_loss(y_test, y_pred).sum()/len(y_test))

    ##### PLOT RESULTS
    if (i) % 100 == 0:
        plot_update(y_train, y_hat, y_test, y_pred, res_train, res_test)

################################################################################
# %% TESTING
################################################################################

##### FOR DEFAULT LAYER PARAMETERS
#assert ffnn.weights['h0'].shape == (2, 2)
#assert ffnn.biases['h0'].shape == (2,)
#assert ffnn.weights['out'].shape == (2, 1)
#assert ffnn.biases['out'].shape == (1,)

##### TEST LOG_LOSS FUNCTION
ytrue = np.array([0.0, 0.0, 1.0, 1.0])
ypred = np.array([0.01, 0.99, 0.01, 0.99])
expected = np.array([0.01, 4.61, 4.61, 0.01])
assert np.all(log_loss(ytrue, ypred).round(2) == expected)

################################################################################
# %% PRINT STUFF
################################################################################

ffnn
