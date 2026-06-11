import numpy as np

EPS = 1e-12

class Dense:
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / in_features)
        self.W = (np.random.randn(in_features, out_features) * scale)
        self.b = np.zeros((1, out_features))

    def forward(self, x):
        self.input = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self.input.T @ grad
        self.db = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T  # dL/dx


class ReLU:
    def forward(self, x):
        self.input = x
        return np.maximum(0.0, x)

    def backward(self, grad):
        return grad * (self.input > 0)


class Softmax:
    def forward(self, x):
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output

    def backward(self, grad):
        return grad


class CrossEntropy:
    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, EPS, 1.0)
        return (-np.sum(y_true * np.log(y_pred)) / y_pred.shape[0])

    def backward(self, y_pred, y_true):
        return (y_pred - y_true) / y_pred.shape[0]


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, layers):
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t

        for layer in layers:
            if not isinstance(layer, Dense):
                continue
            
            for param, grad in ((layer.W, layer.dW), (layer.b, layer.db)):
                key = id(param)

                if key not in self.m:
                    self.m[key] = np.zeros_like(param)
                    self.v[key] = np.zeros_like(param)
                
                m = self.m[key]
                v = self.v[key]

                m[:] = (self.beta1 * m + (1 - self.beta1) * grad)
                v[:] = (self.beta2 * v + (1 - self.beta2) * grad * grad)

                m_hat = m / bc1
                v_hat = v / bc2
                param -= (self.lr * m_hat / (np.sqrt(v_hat) + self.eps))


class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)
        return self

    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def evaluate(self, x, y):
        preds = self.forward(x)
        loss = self.loss.forward(preds, y)
        acc = np.mean(np.argmax(preds, axis=1) == np.argmax(y, axis=1))
        return loss, acc

    def fit(self, X, y, epochs, batch_size, validation_data=None, seed=42):
        rng = np.random.default_rng(seed)
        
        history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
        n = len(X)

        for epoch in range(1, epochs + 1):
            order = rng.permutation(n)

            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                xb, yb = X[idx], y[idx]

                preds = self.forward(xb)
                grad = self.loss.backward(preds, yb)
                self.backward(grad)
                self.optimizer.step(self.layers)

            train_loss, train_acc = self.evaluate(X, y)
            
            history["loss"].append(train_loss)
            history["acc"].append(train_acc)

            if validation_data is not None:
                val_loss, val_acc = self.evaluate(*validation_data)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                print(
                    f"Epoch {epoch:02d}/{epochs} "
                    f"loss={train_loss:.4f} "
                    f"acc={train_acc:.4f} "
                    f"val_loss={val_loss:.4f} "
                    f"val_acc={val_acc:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch:02d}/{epochs} "
                    f"loss={train_loss:.4f} "
                    f"acc={train_acc:.4f}"
                )

        return history