import numpy as np

from datasetloader import load_data
from nn import Network, Dense, ReLU, Softmax, Adam, CrossEntropy

SEED = 42
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

def build_network():
    return (
        Network()
        .add(Dense(784, 256))
        .add(ReLU())
        .add(Dense(256, 128))
        .add(ReLU())
        .add(Dense(128, 10))
        .add(Softmax())
    )

def main():
    np.random.seed(SEED)

    print("Loading MNIST...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(
        val_split=0.1, seed=SEED,
    )
    
    print(
        f"train={len(X_train)} "
        f"val={len(X_val)} "
        f"test={len(X_test)}"
    )

    net = build_network()
    net.compile(loss=CrossEntropy(), optimizer=Adam(lr=LEARNING_RATE))
    net.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        seed=SEED,
    )

    test_loss, test_acc = net.evaluate(X_test, y_test)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

if __name__ == "__main__":
    main()