import json
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from data import load_data
from nn import Adam, Dense, Dropout, Network, ReLU, Sigmoid, SoftmaxCrossEntropy, SGD

FIXED_CONFIG = {
    "architecture": [256, 128],
    "activation": "ReLU",
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 10,
    "seed": 42,
}

DROPOUT_RATES = [0.0, 0.1, 0.2, 0.3, 0.5]


def build_model(dropout_rate, config):
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

        if dropout_rate > 0.0:
            net.add(Dropout(rate=dropout_rate, seed=seed))

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
    print(f"  Training completed in {training_time:.2f} seconds.")
    return history, training_time


def evaluate_model(net, x_test, y_test):
    test_loss, test_acc = net.evaluate(x_test, y_test)
    print(f"  Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    return test_loss, test_acc


def main():
    save_dir = Path(__file__).resolve().parent
    np.random.seed(FIXED_CONFIG["seed"])

    print("Loading MNIST dataset...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(
        seed=FIXED_CONFIG["seed"]
    )

    results = []

    for dropout_rate in DROPOUT_RATES:
        print(f"\n--- Dropout Rate: {dropout_rate} ---")
        model = build_model(dropout_rate, FIXED_CONFIG)
        history, training_time = train_model(
            model, x_train, y_train, x_val, y_val, FIXED_CONFIG
        )
        test_loss, test_acc = evaluate_model(model, x_test, y_test)

        train_loss = float(history["loss"][-1])
        train_acc = float(history["acc"][-1])
        val_loss = float(history["val_loss"][-1])
        val_acc = float(history["val_acc"][-1])
        best_val_acc = float(max(history["val_acc"]))
        best_val_loss = float(min(history["val_loss"]))
        best_epoch = int(np.argmax(history["val_acc"]) + 1)

        result = {
            "dropout_rate": dropout_rate,
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc,
            "test_accuracy": float(test_acc),
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "test_loss": float(test_loss),
            "best_validation_accuracy": best_val_acc,
            "best_validation_loss": best_val_loss,
            "best_epoch": best_epoch,
            "training_time_seconds": training_time,
        }
        results.append(result)

    best = max(results, key=lambda x: (
        x["best_validation_accuracy"],
        -x["best_validation_loss"],
    ))

    metrics = {
        "experiment": "dropout_ablation",
        "fixed_configuration": {
            "architecture": FIXED_CONFIG["architecture"],
            "optimizer": FIXED_CONFIG["optimizer"],
            "learning_rate": FIXED_CONFIG["learning_rate"],
            "epochs": FIXED_CONFIG["epochs"],
            "batch_size": FIXED_CONFIG["batch_size"],
            "activation": FIXED_CONFIG["activation"],
            "seed": FIXED_CONFIG["seed"],
        },
        "selected_dropout": best["dropout_rate"],
        "results": results,
    }

    metrics_path = save_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved experiment metrics to {metrics_path}")

    rates = [str(r["dropout_rate"]) for r in results]
    val_accs = [r["best_validation_accuracy"] for r in results]
    val_losses = [r["best_validation_loss"] for r in results]
    x = np.arange(len(rates))

    plots_dir = save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(x, val_accs, color="royalblue")
    plt.xticks(x, rates)
    plt.xlabel("Dropout Rate")
    plt.ylabel("Best Validation Accuracy")
    plt.title("Dropout Rate vs Best Validation Accuracy")
    for i, v in enumerate(val_accs):
        plt.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
    plt.grid(True, axis="y", alpha=0.3)
    acc_plot_path = plots_dir / "accuracy_vs_dropout.png"
    plt.tight_layout()
    plt.savefig(acc_plot_path, dpi=300)
    plt.close()
    print(f"Saved plot to {acc_plot_path}")

    plt.figure(figsize=(8, 5))
    plt.bar(x, val_losses, color="coral")
    plt.xticks(x, rates)
    plt.xlabel("Dropout Rate")
    plt.ylabel("Best Validation Loss")
    plt.title("Dropout Rate vs Best Validation Loss")
    for i, v in enumerate(val_losses):
        plt.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
    plt.grid(True, axis="y", alpha=0.3)
    loss_plot_path = plots_dir / "loss_vs_dropout.png"
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"Saved plot to {loss_plot_path}")

    print("\n" + "=" * 25)
    print(f"Selected Dropout : {best['dropout_rate']}")
    print(f"Val Accuracy     : {best['best_validation_accuracy'] * 100:.2f}%")
    print(f"Test Accuracy    : {best['test_accuracy'] * 100:.2f}%")
    print("=" * 25)


if __name__ == "__main__":
    main()
