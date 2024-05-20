import numpy as np
from projet_etu import Loss
class MSELoss(Loss):
    """

    """
    def forward(self, y, yhat):
        return np.linalg.norm(y - yhat, axis=1) ** 2

    def backward(self, y, yhat):
        return -2 * (y - yhat)

class CELossSoftMax(Loss):
    """

    """
    def forward(self, y, yhat):
        return (-yhat[np.arange(yhat.shape[0]), y] + np.log(np.exp(yhat).sum(axis=1))).mean()

    def backward(self, y, yhat):
        p = np.exp(yhat) / np.exp(yhat).sum(axis=1).reshape(yhat.shape[0], 1)
        p[np.arange(yhat.shape[0]), y] += -1
        return p / y.shape[0]


class BCELoss(Loss):
    """

    """
    def forward(self, y, yhat):
        threshold = 1e-12
        yhat_threshold = np.clip(yhat, threshold, 1 - threshold)

        return (-(y * np.log(yhat_threshold) + (1 - y) * np.log(1 - yhat_threshold))).mean()

    def backward(self, y, yhat):
        threshold = 1e-12
        yhat_threshold = np.clip(yhat, threshold, 1 - threshold)

        return (yhat_threshold - y) / (yhat_threshold * (1 - yhat_threshold) * yhat.shape[0])