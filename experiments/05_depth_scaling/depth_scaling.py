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
    "activation": "ReLU",
    "dropout": 0.0,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 10,
    "seed": 42,
}

ARCHITECTURES = [
    [128],
    [128, 128],
    [128, 128, 128],
    [128, 128, 128, 128],
    [128, 128, 128, 128, 128],
]


def build_model(architecture, config):
    net = Network()
    seed = config["seed"]
    in_dim = 784

    for out_dim in architecture:
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
    print(f"  Training completed in {training_time:.2f} seconds.")
    return history, training_time


def evaluate_model(net, x_test, y_test):
    test_loss, test_acc = net.evaluate(x_test, y_test)
    print(f"  Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    return test_loss, test_acc


def count_parameters(net):
    num = 0
    for layer in net.layers:
        for param, _ in layer.params_and_grads():
            num += param.size
    return num


def main():
    save_dir = Path(__file__).resolve().parent
    np.random.seed(FIXED_CONFIG["seed"])

    print("Loading MNIST dataset...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(
        seed=FIXED_CONFIG["seed"]
    )

    results = []

    for architecture in ARCHITECTURES:
        depth = len(architecture)
        print(f"\n--- Depth: {depth} ---")
        model = build_model(architecture, FIXED_CONFIG)
        num_params = count_parameters(model)
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
            "architecture": architecture,
            "depth": depth,
            "num_parameters": num_params,
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

    best = max(results, key=lambda x: x["best_validation_accuracy"])

    metrics = {
        "experiment": "depth_scaling",
        "fixed_configuration": {
            "hidden_width": 128,
            "activation": FIXED_CONFIG["activation"],
            "optimizer": FIXED_CONFIG["optimizer"],
            "learning_rate": FIXED_CONFIG["learning_rate"],
            "epochs": FIXED_CONFIG["epochs"],
            "batch_size": FIXED_CONFIG["batch_size"],
            "seed": FIXED_CONFIG["seed"],
        },
        "selected_depth": best["depth"],
        "results": results,
    }

    metrics_path = save_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved experiment metrics to {metrics_path}")

    depths = [r["depth"] for r in results]
    val_accs = [r["best_validation_accuracy"] for r in results]
    times = [r["training_time_seconds"] for r in results]
    params = [r["num_parameters"] for r in results]
    x = np.arange(len(depths))
    depth_labels = [str(d) for d in depths]

    plots_dir = save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(x, val_accs, color="royalblue")
    plt.xticks(x, depth_labels)
    plt.ylabel("Best Validation Accuracy")
    plt.title("Depth vs Best Validation Accuracy")
    for i, v in enumerate(val_accs):
        plt.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
    plt.grid(True, axis="y", alpha=0.3)
    acc_plot_path = plots_dir / "accuracy_vs_depth.png"
    plt.tight_layout()
    plt.savefig(acc_plot_path, dpi=300)
    plt.close()
    print(f"Saved plot to {acc_plot_path}")

    plt.figure(figsize=(8, 5))
    plt.bar(x, times, color="coral")
    plt.xticks(x, depth_labels)
    plt.ylabel("Training Time (s)")
    plt.title("Depth vs Training Time")
    for i, v in enumerate(times):
        plt.text(i, v + 0.5, f"{v:.1f}s", ha="center", fontsize=9)
    plt.grid(True, axis="y", alpha=0.3)
    time_plot_path = plots_dir / "time_vs_depth.png"
    plt.tight_layout()
    plt.savefig(time_plot_path, dpi=300)
    plt.close()
    print(f"Saved plot to {time_plot_path}")

    plt.figure(figsize=(8, 5))
    plt.scatter(params, val_accs, c="green", s=80)
    for i, label in enumerate(depth_labels):
        plt.text(params[i] + 5000, val_accs[i], f"Depth {label}", fontsize=9)
    plt.xlabel("Number of Parameters")
    plt.ylabel("Best Validation Accuracy")
    plt.title("Parameters vs Best Validation Accuracy")
    plt.grid(True, alpha=0.3)
    scatter_plot_path = plots_dir / "accuracy_vs_parameters.png"
    plt.tight_layout()
    plt.savefig(scatter_plot_path, dpi=300)
    plt.close()
    print(f"Saved plot to {scatter_plot_path}")

    print("\n" + "=" * 30)
    print(f"Selected Depth : {best['depth']}")
    print(f"Architecture   : {'-'.join(str(d) for d in best['architecture'])}")
    print(f"Parameters     : {best['num_parameters']}")
    print(f"Val Accuracy   : {best['best_validation_accuracy'] * 100:.2f}%")
    print(f"Test Accuracy  : {best['test_accuracy'] * 100:.2f}%")
    print("=" * 30)


if __name__ == "__main__":
    main()
