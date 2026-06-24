import numpy as np
from collections.abc import Iterator

class Layer:   
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, x, training=False):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def params_and_grads(self) -> Iterator:
        yield from ()


class Dense(Layer):   
    def __init__(self, in_features, out_features, weight_init="he", seed=None):
        super().__init__()
        rng = np.random.default_rng(seed)

        if weight_init == "he":
            scale = np.sqrt(2.0 / in_features)
        elif weight_init == "xavier":
            scale = np.sqrt(1.0 / in_features)
        else:
            raise ValueError(f"unknown weight_init: {weight_init!r}")

        self.W = rng.normal(0.0, scale, size=(in_features, out_features))
        self.b = np.zeros((1, out_features))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x, training=False):
        self.input = x
        self.output = x @ self.W + self.b
        return self.output

    def backward(self, grad_output):
        self.dW = self.input.T @ grad_output
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.W.T  # dL/dx

    def params_and_grads(self):
        yield self.W, self.dW
        yield self.b, self.db


class ReLU(Layer):
    def forward(self, x, training=False):
        self.input = x
        self.output = np.maximum(0.0, x)
        return self.output

    def backward(self, grad_output):
        return grad_output * (self.input > 0)


class Sigmoid(Layer):
    def forward(self, x, training=False):
        self.input = x
        self.output = np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
            np.exp(np.clip(x, -500, 500)) / (1.0 + np.exp(np.clip(x, -500, 500))),
        )
        return self.output

    def backward(self, grad_output):
        s = self.output
        return grad_output * s * (1.0 - s)


class Softmax(Layer):
    def forward(self, x, training=False):
        self.input = x
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        s = self.output
        dot = np.sum(grad_output * s, axis=1, keepdims=True)
        return s * (grad_output - dot)


class Dropout(Layer):
    def __init__(self, rate=0.5, seed=None):
        super().__init__()
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"dropout rate must be in [0, 1), got {rate}")
        self.rate = rate
        self.keep = 1.0 - rate
        self.mask = None
        self._rng = np.random.default_rng(seed)

    def forward(self, x, training=False):
        self.input = x
        if not training or self.rate == 0.0:
            self.mask = None
            self.output = x
            return x
        self.mask = (self._rng.random(x.shape) < self.keep) / self.keep
        self.output = x * self.mask
        return self.output

    def backward(self, grad_output):
        if self.mask is None:
            return grad_output
        return grad_output * self.mask