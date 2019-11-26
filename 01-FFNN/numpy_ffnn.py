"""
Function and class helpers to run a FFNN for classification built entirely on numpy
"""

import numpy as np
import matplotlib.pyplot as mp
from IPython import display

################################################################################
# %% DEFINE FUNCTIONS / CLASS / GENERATOR
################################################################################

def sigmoid(X):
    """Signmoid activation function"""
    return 1.0/(1.0+np.exp(-X))

def sigmoid_derivative(X):
    """Sigmoid drivative function"""
    return sigmoid(X)*(1.0-sigmoid(X))

def log_loss(y_true, y_hat):
    """Calculate the logarithmic loss, element-wise"""
    return -(y_true*np.log(y_hat) + (1.0-y_true)*np.log(1.0-y_hat))

def error(y_true, y_hat):
    """Calculate error between output layer and true result"""
    return -(y_true - y_hat)*log_loss(y_true, y_hat)

def accuracy(y_true, y_hat):
    """Calculate classification accurancy"""
    return (y_true.astype(float) == y_hat.round()).mean()

def plot_update(y_train, y_hat, y_test, y_pred, res_train, res_test):
    """Update learning curve plot"""
    display.clear_output(wait=True)
    mp.figure()
    mp.title(f'Accuracy: Train: {accuracy(y_train,y_hat)}, Test: {accuracy(y_test,y_pred)}')
    mp.semilogy(res_train,label=f"train {res_train[-1]:.2f}")
    mp.semilogy(res_test,label=f"test {res_test[-1]:.2f}")
    mp.legend()
    mp.xlabel('Epochs')
    mp.ylabel('Loss')
    mp.show()

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
        self.learnrates = {'h0': 1e-1, 'out': 5e0}
        self.init_func = 'random'

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
        if self.init_func == 'ones':
            self.weights[current_layer] = np.ones((X.shape[1],self.nodes[current_layer]))
            self.biases[current_layer] = np.ones((self.nodes[current_layer]))
        elif self.init_func == 'random':
            self.weights[current_layer] = np.random.random(size=(X.shape[1],self.nodes[current_layer]))
            self.biases[current_layer] = np.random.random(size=(self.nodes[current_layer]))

        ##### HIDDEN TO HIDDEN LAYERS
        for layer in range(1,self.hlayers):
            last_layer = 'h'+str(layer-1)
            current_layer = 'h'+str(layer)
            if self.init_func == 'ones':
                self.weights[current_layer] = np.ones((self.nodes[last_layer],self.nodes[current_layer]))
                self.biases[current_layer] = np.ones((self.nodes[current_layer]))
            elif self.init_func == 'random':
                self.weights[current_layer] = np.random.random(size=(self.nodes[last_layer],self.nodes[current_layer]))
                self.biases[current_layer] = np.random.random(size=(self.nodes[current_layer]))

        ##### HIDDEN TO OUTPUT LAYER
        last_layer = 'h'+str(self.hlayers-1)
        current_layer = 'out'
        if self.init_func == 'ones':
            self.weights[current_layer] = np.ones((self.nodes[last_layer],y.shape[1]))
            self.biases[current_layer] = np.ones((y.shape[1]))
        elif self.init_func == 'random':
            self.weights[current_layer] = np.random.random(size=(self.nodes[last_layer],y.shape[1]))
            self.biases[current_layer] = np.random.random(size=(y.shape[1]))


    def forward(self,X,y,train):
        """Run a feed-forward step"""
        ##### IF RUNNING FOR FIRST TIME INIT RANDOM WEIGHTS AND BIASES
        if self.weights == None or self.biases == None:
            self.build(X,y)

        if train == 'train':
            self.outputs = {}
        ##### RUN THROUGH ALL HIDDEN LAYERS
        X_in = X
        for layer in range(self.hlayers):
            layer_name = 'h'+str(layer)
            out = sigmoid(np.dot(X_in, self.weights[layer_name]) + self.biases[layer_name])
            X_in = out
            if train == 'train':
                self.outputs[layer_name] = out

        ##### RUN THROUGH OUTPUT LAYER
        layer_name = 'out'
        out = sigmoid(np.dot(X_in, self.weights[layer_name]) + self.biases[layer_name])
        if train == 'train':
            self.outputs[layer_name] = out
        return out

    def backprop_layer(self, input, weights, biases, err, LR):
        """Calculate deltas for backpropagation over a single layer"""
        Dy = sigmoid_derivative(
            np.dot(
                input,
                weights
            )+biases)*err

        dw = -(np.dot(Dy.T, input))*LR
        db = -(Dy.sum())*LR

        return dw, db, Dy

    def backpropagate(self, X, y_true, y_hat):
        """Calculate backpropagation over all layers"""
        ##### HIDDEN TO OUTPUT LAYER
        ##### CALCULATE ERROR
        err = error(y_true, y_hat)

        ##### GET/APPLY CORRECTIONS
        last_layer = 'h'+str(self.hlayers-1)
        dw, db, Dy = self.backprop_layer(self.outputs[last_layer], self.weights['out'], self.biases['out'], err, self.learnrates['out'])
        self.weights['out'] = self.weights['out'] + dw.T
        self.biases['out'] = self.biases['out'] + db

        ##### HIDDEN TO HIDDEN LAYERS
        for layer in reversed(list(range(1,self.hlayers))):
            last_layer = 'h'+str(layer-1)
            current_layer = 'h'+str(layer)
            if layer+1 == self.hlayers:
                next_layer = 'out'
            else:
                next_layer = 'h'+str(layer+1)

            ##### CALCULATE ERROR
            err = np.dot(Dy, self.weights[next_layer].T)

            ##### GET/APPLY CORRECTIONS
            dw, db, Dy = self.backprop_layer(self.outputs[last_layer], self.weights[current_layer], self.biases[current_layer], err, self.learnrates[current_layer])
            self.weights[current_layer] = self.weights[current_layer] + dw.T
            self.biases[current_layer] = self.biases[current_layer] + db


        ##### INPUT TO HIDDEN LAYERS
        ##### CALCULATE ERROR
        err = np.dot(Dy, self.weights['h1'].T)

        ##### GET/APPLY CORRECTIONS
        dw, db, Dy = self.backprop_layer(X, self.weights['h0'], self.biases['h0'], err, self.learnrates['h0'])
        self.weights['h0'] = self.weights['h0'] + dw.T
        self.biases['h0'] = self.biases['h0'] + db

        return log_loss(y_true, y_hat).sum()
