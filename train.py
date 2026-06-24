import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data.datasetloader import load_data
from nn.nn import Network, Dense, ReLU, Softmax, Adam, CrossEntropy

SEED = 42
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PLOT_PATH = "assets/plot.png"

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

def plot_history(history, output_path=PLOT_PATH):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(12, 4))

    loss_ax.plot(epochs, history["loss"], label="Train loss")
    if history["val_loss"]:
        loss_ax.plot(epochs, history["val_loss"], label="Val loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Cross-entropy loss")
    loss_ax.set_title("Loss")
    loss_ax.legend()
    loss_ax.grid(True, alpha=0.3)

    acc_ax.plot(epochs, history["acc"], label="Train accuracy")
    if history["val_acc"]:
        acc_ax.plot(epochs, history["val_acc"], label="Val accuracy")
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy")
    acc_ax.set_title("Accuracy")
    acc_ax.legend()
    acc_ax.grid(True, alpha=0.3)

    fig.suptitle("MNIST Neural Network")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def main():
    np.random.seed(SEED)

    print("Loading MNIST...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(
        val_split=0.1, seed=SEED,
    )
    
    print(f"train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    net = build_network()
    net.compile(loss=CrossEntropy(), optimizer=Adam(lr=LEARNING_RATE))
    
    history = net.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        seed=SEED,
    )

    plot_history(history)

    test_loss, test_acc = net.evaluate(X_test, y_test)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

if __name__ == "__main__":
    main()