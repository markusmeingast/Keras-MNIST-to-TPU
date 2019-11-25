"""

"""

################################################################################
# %% IMPORT PACKAGES
################################################################################

import numpy as np
import matplotlib.pyplot as mp

################################################################################
# %% DEFINE FUNCTIONS / CLASS / GENERATOR
################################################################################

def sigmoid(X):
    return 1.0/(1.0+np.exp(-X))

def sigmoid_derivative(X):
    return sigmoid(X)*(1.0-sigmoid(X))

class FFNN():
    """
    Feed Formward class with variable layers and nodes per layer.
    """

    def __init__(self):
        """Set standard paramteres"""
        ##### DEFAULT NUMBER OF HIDDEN LAYERS
        self.hlayers = 1
        ##### DEFAULT NUMBER OF NODES PER LAYER
        self.nodes = {'h0' : 2}
        self.weights = None
        self.biases = None
        self.learnrate = 1e-1

    def __repr__(self):
        """Default representation"""
        string = """"""
        string += f"""FFNN Model: \nNumber of hidden layers: {self.hlayers}\n"""
        for layer in self.nodes:
                string += f"""Number of nodes in layer {layer} is {self.nodes[layer]}\n"""
        return string

    def build(self,X,y):
        """Initialize random weights and biases"""
        self.weights = {}
        self.biases = {}

        ##### INPUT TO FIRST HIDDEN LAYER
        current_layer = 'h0'
        self.weights[current_layer] = np.random.random(size=(X.shape[1],self.nodes[current_layer]))
        self.biases[current_layer] = np.random.random(size=(self.nodes[current_layer]))

        ##### HIDDEN TO HIDDEN LAYERS
        for layer in range(1,self.hlayers):
            last_layer = 'h'+str(layer-1)
            current_layer = 'h'+str(layer)
            self.weights[current_layer] = np.random.random(size=(self.nodes[last_layer],self.nodes[current_layer]))
            self.biases[current_layer] = np.random.random(size=(self.nodes[current_layer]))

        ##### HIDDEN TO OUTPUT LAYER
        last_layer = 'h'+str(self.hlayers-1)
        current_layer = 'out'
        self.weights[current_layer] = np.random.random(size=(self.nodes[last_layer],y.shape[1]))
        self.biases[current_layer] = np.random.random(size=(y.shape[1]))

    def forward(self,X,y):
        """Run a feed-forward step"""
        ##### IF RUNNING FOR FIRST TIME INIT RANDOM WEIGHTS AND BIASES
        if self.weights == None or self.biases == None:
            self.build(X,y)

        self.outputs = {}
        ##### RUN THROUGH ALL HIDDEN LAYERS
        X_in = X
        for layer in range(self.hlayers):
            layer_name = 'h'+str(layer)
            out = sigmoid(np.dot(X_in, self.weights[layer_name]) + self.biases[layer_name])
            X_in = out
            self.outputs[layer_name] = out

        ##### RUN THROUGH OUTPUT LAYER
        layer_name = 'out'
        out = sigmoid(np.dot(X_in, self.weights[layer_name]) + self.biases[layer_name])
        self.outputs[layer_name] = out
        return out

    def log_loss(self, y_true, y_hat):
        loss = -(y_true*np.log(y_hat) + (1.0-y_true)*np.log(1.0-y_hat))
        return loss

    def error(self,y_true,y_hat):
        """Calculate error between output layer and true result"""
        return -(y_true - y_hat)*self.log_loss(y_true, y_hat)

    def backpropagate(self, X, y_true, y_hat):
        ##### A
        err = self.error(y_true, y_hat)

        ##### B
        #Dy = sigmoid_derivative(np.dot(self.outputs['h0'], self.weights['out']))*err
        #Db = sigmoid_derivative(self.biases['out'])*err

        Dy = sigmoid_derivative(
            np.dot(
                np.hstack((self.outputs['h0'],np.ones((50,1)))),
                np.vstack((self.weights['out'],self.biases['out']))
                ))*err


        ##### C
        #dw = -(np.dot(Dy.T, self.outputs['out']))*0.1
        dw = -(np.dot(Dy.T, np.hstack((self.outputs['h0'],np.ones((50,1)))))).T*0.1
        #db = -(Db.sum())*0.1

        self.weights['out'] = self.weights['out'] + dw[:2,:]
        self.biases['out'] = self.biases['out'] + dw[2,:]


        err = np.dot(Dy, self.weights['out'].T)

        #print(err.shape)

        ##### D
        #Dy = sigmoid_derivative(np.dot(X, self.weights['h0']))*err
        #Db = sigmoid_derivative(self.biases['h0'])*err
        Dy = sigmoid_derivative(
            np.dot(
                np.hstack((X,np.ones((50,1)))),
                np.vstack((self.weights['h0'],self.biases['h0']))
                ))*err

        #print(Dy.shape)

        #print(Dy.shape)
        #####
        #dw = -(np.dot(Dy.T, self.outputs['h0']))*5.0
        dw = -(np.dot(Dy.T, np.hstack((X,np.ones((50,1)))))).T*5.0
        #db = -(Db.sum())*5.0
        #print(dw.shape)
        self.weights['h0'] = self.weights['h0'] + dw[:2,:]
        self.biases['h0'] = self.biases['h0'] + dw[2,:]

        return self.log_loss(y_true, y_hat).sum()

################################################################################
# %% IMPORT DATASET
################################################################################

from sklearn.datasets import make_moons
X, y = make_moons(n_samples=50, noise=0.0, random_state=42)
y = y.reshape(-1,1)

################################################################################
# %% INIT AND BUILD FFNN
################################################################################

ffnn = FFNN()

################################################################################
# %% SET PARAMETERS IF REQUIRED
################################################################################

#ffnn.hlayers = 3
#ffnn.nodes = {'h0':5, 'h1':3, 'h2':4}

################################################################################
# %% RUN FORWARD FEED
################################################################################

for i in range(500):
    y_hat = ffnn.forward(X,y)
    loss = ffnn.backpropagate(X, y, y_hat)
    print(loss)

################################################################################
# %% PRINT STUFF
################################################################################

mp.plot(y,y_hat.round())

ffnn
ffnn.weights
ffnn.biases


y.shape

ffnn.error(y_hat,y)

################################################################################
# %% TESTING
################################################################################

#####

assert ffnn.weights['h0'].shape == (2, 2)
assert ffnn.biases['h0'].shape == (2,)
assert ffnn.weights['out'].shape == (2, 1)
assert ffnn.biases['out'].shape == (1,)

#####

ytrue = np.array([0.0, 0.0, 1.0, 1.0])
ypred = np.array([0.01, 0.99, 0.01, 0.99])
expected = np.array([0.01, 4.61, 4.61, 0.01])
assert np.all(ffnn.log_loss(ytrue, ypred).round(2) == expected)
