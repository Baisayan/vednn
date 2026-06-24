import numpy as np

_EPS = 1e-12

class Loss:

    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self, y_pred, y_true):
        raise NotImplementedError

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

class MSE(Loss):

    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        n = y_pred.shape[0] * y_pred.shape[1]
        return 2.0 * (y_pred - y_true) / n


class CrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        n = y_pred.shape[0]
        clipped = np.clip(y_pred, _EPS, 1.0)
        return -np.sum(y_true * np.log(clipped)) / n

    def backward(self, y_pred, y_true):
        n = y_pred.shape[0]
        clipped = np.clip(y_pred, _EPS, 1.0)
        return -(y_true / clipped) / n