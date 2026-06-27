import json
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Adjust python path to project root
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from data import load_data
from nn import SGD, Adam, Dense, Dropout, Network, ReLU, Sigmoid, SoftmaxCrossEntropy

CONFIG = {
    "dataset": "MNIST",
    "seed": 42,
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 0.001,
    "architecture": [256, 128],
    "activation": "ReLU",
    "dropout": 0.0,
    "optimizer": "Adam",
}


def build_model(config):
    net = Network()
    seed = config["seed"]
    in_dim = 784

    for out_dim in config["architecture"]:
        net.add(Dense(in_dim, out_dim, seed=seed))

        if config["activation"] == "ReLU":
            net.add(ReLU())
        elif config["activation"] == "Sigmoid":
            net.add(Sigmoid())
        else:
            raise ValueError(f"Unknown activation type: {config['activation']}")

        if config["dropout"] > 0.0:
            net.add(Dropout(rate=config["dropout"], seed=seed))

        in_dim = out_dim

    net.add(Dense(in_dim, 10, seed=seed))
    loss_fn = SoftmaxCrossEntropy()

    if config["optimizer"] == "Adam":
        optimizer = Adam(lr=config["learning_rate"])
    elif config["optimizer"] == "SGD":
        optimizer = SGD(lr=config["learning_rate"])
    else:
        raise ValueError(f"Unknown optimizer type: {config['optimizer']}")

    net.compile(loss=loss_fn, optimizer=optimizer)
    return net


def train_model(net, x_train, y_train, x_val, y_val, config):
    start_time = time.perf_counter()
    history = net.fit(
        x_train,
        y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        validation_data=(x_val, y_val),
        seed=config["seed"],
    )
    training_time = time.perf_counter() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    return history, training_time


def evaluate_model(net, x_test, y_test):
    test_loss, test_acc = net.evaluate(x_test, y_test)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    return test_loss, test_acc


def count_parameters(net):
    num_parameters = 0
    for layer in net.layers:
        for param, _ in layer.params_and_grads():
            num_parameters += param.size
    return num_parameters


def save_metrics(config, history, test_loss, test_acc, training_time, num_parameters, save_dir):
    train_loss_hist = [float(x) for x in history["loss"]]
    train_acc_hist = [float(x) for x in history["acc"]]
    val_loss_hist = (
        [float(x) for x in history["val_loss"]] if "val_loss" in history else []
    )
    val_acc_hist = (
        [float(x) for x in history["val_acc"]] if "val_acc" in history else []
    )

    metrics = {
        "dataset": config["dataset"],
        "architecture": config["architecture"],
        "activation": config["activation"],
        "dropout": config["dropout"],
        "optimizer": config["optimizer"],
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "seed": config["seed"],
        "train_loss": train_loss_hist[-1] if train_loss_hist else None,
        "train_accuracy": train_acc_hist[-1] if train_acc_hist else None,
        "val_loss": val_loss_hist[-1] if val_loss_hist else None,
        "val_accuracy": val_acc_hist[-1] if val_acc_hist else None,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "best_val_accuracy": max(history["val_acc"]),
        "best_val_loss": min(history["val_loss"]),
        "best_epoch": int(np.argmax(history["val_acc"]) + 1),
        "training_time_seconds": float(training_time),
        "num_parameters": int(num_parameters),
        "history": {
            "train_loss": train_loss_hist,
            "train_accuracy": train_acc_hist,
            "val_loss": val_loss_hist,
            "val_accuracy": val_acc_hist,
        },
    }

    metrics_path = save_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved experiment metrics to {metrics_path}")


def plot_history(history, plots_dir):
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = len(history["loss"])
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["loss"], "b-", label="Train Loss")
    if "val_loss" in history and history["val_loss"]:
        plt.plot(epochs_range, history["val_loss"], "r-", label="Val Loss")
    plt.title("MNIST Baseline Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_plot_path = plots_dir / "loss.png"
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["acc"], "b-", label="Train Accuracy")
    if "val_acc" in history and history["val_acc"]:
        plt.plot(epochs_range, history["val_acc"], "r-", label="Val Accuracy")
    plt.title("MNIST Baseline Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    acc_plot_path = plots_dir / "accuracy.png"
    plt.tight_layout()
    plt.savefig(acc_plot_path, dpi=300)
    plt.close()

    print(f"Saved training curves to {loss_plot_path} and {acc_plot_path}")


def main():
    save_dir = Path(__file__).resolve().parent
    np.random.seed(CONFIG["seed"])

    print(f"Loading {CONFIG['dataset']} dataset...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(
        seed=CONFIG["seed"]
    )

    model = build_model(CONFIG)
    history, training_time = train_model(model, x_train, y_train, x_val, y_val, CONFIG)
    test_loss, test_acc = evaluate_model(model, x_test, y_test)
    num_params = count_parameters(model)

    plots_dir = save_dir / "plots"

    save_metrics(
        CONFIG, history, test_loss, test_acc, training_time, num_params, save_dir
    )
    plot_history(history, plots_dir)


if __name__ == "__main__":
    main()
