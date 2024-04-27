import numpy as np
from projet_etu import Loss, Module


class MSELoss(Loss):
    def forward(self, y, yhat):
        return np.linalg.norm(y - yhat, axis=1) ** 2

    def backward(self, y, yhat):
        return -2 * (y - yhat)


class Linear(Module):
    def __init__(self, input, output):
        super().__init__()
        self._parameters = np.random.randn(input, output)
        self._gradient = 0

    def zero_grad(self):
        self._gradient = 0.0

    def forward(self, X):
        return X @ self._parameters

    def backward_update_gradient(self, X, delta):
        self._gradient += X.T @ delta

    def backward_delta(self, X, delta):
        return delta @ self._parameters.T
