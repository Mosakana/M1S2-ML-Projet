import numpy as np
from projet_etu import Loss
class MSELoss(Loss):
    def forward(self, y, yhat):
        return np.linalg.norm(y - yhat, axis=1) ** 2

    def backward(self, y, yhat):
        return -2 * (y - yhat)