import numpy as np

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)
        return self

    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, x, training=False):
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def evaluate(self, x, y):
        if self.loss is None:
            raise RuntimeError("call compile() before fit()")
        
        preds = self.forward(x)
        loss = self.loss.forward(preds, y)
        acc = np.mean(np.argmax(preds, axis=1) == np.argmax(y, axis=1))
        return loss, acc

    def fit(self, X, y, epochs, batch_size, validation_data=None, shuffle=True, seed=None):
        if self.loss is None or self.optimizer is None:
            raise RuntimeError("call compile() before fit()")

        rng = np.random.default_rng(seed)
        n = len(X)
        history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            order = rng.permutation(n) if shuffle else np.arange(n)

            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                xb, yb = X[idx], y[idx]

                preds = self.forward(xb, training=True)
                self.loss(preds, yb)
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