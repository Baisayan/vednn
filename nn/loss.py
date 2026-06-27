import numpy as np

class Loss:
    def forward(self, preds, targets):
        raise NotImplementedError

    def backward(self, preds, targets):
        raise NotImplementedError

    def __call__(self, preds, targets):
        return self.forward(preds, targets)


class MSE(Loss):
    def forward(self, preds, targets):
        return np.mean((preds - targets) ** 2)

    def backward(self, preds, targets):
        return 2.0 * (preds - targets) / preds.size


class SoftmaxCrossEntropy(Loss):
    def __init__(self):
        self.probs = None

    def forward(self, preds, targets):
        shifted = preds - np.max(preds, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)

        n = preds.shape[0]
        clipped = np.clip(self.probs, 1e-12, 1.0)
        return -np.sum(targets * np.log(clipped)) / n

    def backward(self, preds, targets):
        n = preds.shape[0]
        return (self.probs - targets) / n