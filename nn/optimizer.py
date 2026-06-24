import numpy as np

class Optimizer:
    def step(self, layers):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self._velocity = {}

    def step(self, layers):
        for layer in layers:
            for param, grad in layer.params_and_grads():
                if self.momentum:
                    key = id(param)
                    v = self._velocity.get(key)
                    if v is None:
                        v = np.zeros_like(param)
                        self._velocity[key] = v
                    v *= self.momentum
                    v -= self.lr * grad
                    param += v
                else:
                    param -= self.lr * grad

class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self._m = {}
        self._v = {}

    def step(self, layers):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t

        for layer in layers:
            for param, grad in layer.params_and_grads():
                key = id(param)
                m = self._m.get(key)
                if m is None:
                    m = np.zeros_like(param)
                    v = np.zeros_like(param)
                    self._m[key] = m
                    self._v[key] = v
                else:
                    v = self._v[key]

                m *= b1
                m += (1.0 - b1) * grad
                v *= b2
                v += (1.0 - b2) * (grad * grad)

                m_hat = m / bc1
                v_hat = v / bc2
                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)