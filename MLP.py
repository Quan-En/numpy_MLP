
import numpy as np

import utils

class MLP:
    def __init__(self, n_in, n_hidden, n_out):
        # Network dimensions
        self.n_x = n_in
        self.n_h = n_hidden
        self.n_y = n_out

        # Parameters initialization
        self.W1 = np.random.randn(self.n_h, self.n_x) * 0.01
        self.b1 = np.zeros((self.n_h, 1))
        self.W2 = np.random.randn(self.n_y, self.n_h) * 0.01
        self.b2 = np.zeros((self.n_y, 1))

    def forward(self, X):
        """ Forward computation """

        self.Z1 = (self.W1.dot(X.T) + self.b1).T
        self.A1 = utils.tanh(self.Z1)
        self.Z2 = (self.W2.dot(self.A1.T) + self.b2).T
        self.A2 = utils.softmax(self.Z2)

        return self.A2

    def back_prop(self, X, Y):
        """ Back-progagate gradient of the loss """
        m = X.shape[0]

        # dC / dZ2
        self.dZ2 = self.A2 - Y

        # dC / dW2
        self.dW2 = (1 / m) * np.matmul(self.dZ2.T, self.A1)
        
        # dC / db2
        self.db2 = (1 / m) * np.sum(self.dZ2, axis=0, keepdims=True).T

        # dC / dZ1
        self.dZ1 = utils.tanh_derivative(self.A1) * np.matmul(self.dZ2, self.W2)

        # dC / dW1
        self.dW1 = (1 / m) * np.matmul(self.dZ1.T, X)

        # dC / db1
        self.db1 = (1 / m) * np.sum(self.dZ1, axis=0, keepdims=True).T

    def train(self, X, Y, epochs, learning_rate=0.01):
        """
        Complete process of learning,
        alternates forward pass,
        backward pass and parameters update
        """
        m = X.shape[0]
        for e in range(epochs):
            self.forward(X)
            loss = -np.sum(np.sum(Y * np.log(self.A2), axis=1)) / m
            self.back_prop(X, Y)

            self.W2 -= learning_rate * self.dW2
            self.b2 -= learning_rate * self.db2
            self.W1 -= learning_rate * self.dW1
            self.b1 -= learning_rate * self.db1

            if e % 1000 == 0: print("Loss ",  e, " = ", loss)

    def predict(self, X):
        """ Compute predictions with just a forward pass """
        self.forward(X)
        return np.argmax(self.A2, axis=1)