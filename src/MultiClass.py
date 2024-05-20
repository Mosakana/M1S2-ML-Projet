from projet_etu import Loss
import numpy as np

class CELossSoftMax(Loss):
    def forward(self, y, yhat):
        return -yhat[np.arange(yhat.shape[0]), y] + np.log(np.exp(yhat).sum(axis=1))

    def backward(self, y, yhat):
        p = np.exp(yhat) / np.exp(yhat).sum(axis=1).reshape(yhat.shape[0], 1)
        p[np.arange(yhat.shape[0]), y] += -1
        return p