from src.SequentielNet import Sequentiel, Optim
from src.NonLinearModel import Sigmoid, Tanh
from src.LinearModel import Linear
from src.Loss import BCELoss
from copy import deepcopy


import numpy as np
import matplotlib.pyplot as plt

class AutoCoder():
    def __init__(self, *args):
        self.encoder = Sequentiel(*args)
        self.decoder = Sequentiel(*deepcopy(args)[::-1][1:], Sigmoid())

        for model in self.decoder.net:
            if model.__class__.__name__ == 'Linear':
                model._parameters = model._parameters.T

        self.outputs = None

    def forward(self, X):
        self.encoder.forward(X)
        self.decoder.forward(self.encoder.outputs[-1])

        self.outputs = self.decoder.outputs

        return self.decoder.outputs[-1]

    def backward(self, delta, eps):
        self.decoder.backward(delta, eps)
        self.encoder.backward(self.decoder.grads[-1], eps)


class Optim_autocoder():
    def __init__(self, autocoder, loss, eps):
        self.autocoder = autocoder
        self.loss = loss
        self.eps = eps
        self.x_hat = None
        self.score = 0

    def step(self, batch_X):
        self.autocoder.encoder.zero_grad()
        self.autocoder.decoder.zero_grad()

        x_hat = self.autocoder.forward(batch_X)

        self.score += self.loss.forward(batch_X, x_hat)

        self.autocoder.backward(self.loss.backward(batch_X, x_hat), self.eps)

    def update_output(self):
        if self.x_hat is None:
            self.x_hat = self.autocoder.outputs[-1]
        else:
            self.x_hat = np.vstack((self.x_hat, self.autocoder.outputs[-1]))

    def clean_output(self):
        self.x_hat = None

    def zero_score(self):
        self.score = 0


def SGD_autocoder(autocoder, X, batch_size, epochs, loss, eps=1e-3):
    optim = Optim_autocoder(autocoder, loss, eps)
    n_batch = X.shape[0] // batch_size

    for iteration in range(epochs):
        optim.clean_output()
        optim.zero_score()
        for i in range(n_batch):
            optim.step(X[i * batch_size: (i+1) * batch_size])
            optim.update_output()

        optim.step(X[n_batch * batch_size:])
        optim.update_output()

    return optim
