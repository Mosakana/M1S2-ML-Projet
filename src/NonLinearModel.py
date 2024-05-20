import numpy as np
from src.projet_etu import Module


class Tanh(Module):
    def __init__(self):
        super().__init__()
        self._gradient = 0.0

    def forward(self, X):
        return (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))

    def zero_grad(self):
        self._gradient = 0.0

    def backward_delta(self, X, delta):
        return delta * ((1 / (np.cosh(X))) ** 2)

    def backward_update_gradient(self, X, delta):
        pass

    def update_parameters(self, gradient_step=1e-3):
        pass


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self._gradient = 0.0

    def forward(self, X):
        return 1 / (1 + np.exp(-X))

    def zero_grad(self):
        self._gradient = 0.0

    def backward_delta(self, X, delta):
        return delta * (np.exp(-X) / ((np.exp(-X) + 1) ** 2))

    def backward_update_gradient(self, X, delta):
        pass

    def update_parameters(self, gradient_step=1e-3):
        pass


class LogSoftMax(Module):
    """

    """
    def __init__(self):
        super().__init__()
        self._gradient = 0.0

    def forward(self, X):
        return np.log(np.exp(X) / np.exp(X).sum())

    def zero_grad(self):
        self._gradient = 0.0

    def backward_delta(self, X, delta):
        p = np.exp(X) / np.exp(X).sum()
        length_p = p.shape[0]
        Jacobian = np.tile(-p, (length_p, 1))
        for i in range(length_p):
            Jacobian[i, i] = 1 + Jacobian[i, i]

        return Jacobian.T @ delta

    def backward_update_gradient(self, X, delta):
        pass

    def update_parameters(self, gradient_step=1e-3):
        pass
