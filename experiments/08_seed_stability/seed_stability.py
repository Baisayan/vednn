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
    "dropout": 0.2,
    "weight_init": "xavier",
    "optimizer": "Adam",
    "learning_rate": 0.0003,
    "batch_size": 64,
    "epochs": 20,
}

SEEDS = [0, 21, 42, 123, 999]


def build_model(seed, config):
    net = Network()
    in_dim = 784

    for out_dim in config["architecture"]:
        net.add(Dense(in_dim, out_dim, weight_init=config["weight_init"], seed=seed))

        if config["activation"] == "ReLU":
            net.add(ReLU())
        elif config["activation"] == "Sigmoid":
            net.add(Sigmoid())
        else:
            raise ValueError(f"Unknown activation type: {config['activation']}")

        if config["dropout"] > 0.0:
            net.add(Dropout(rate=config["dropout"], seed=seed))

        in_dim = out_dim

    net.add(Dense(in_dim, 10, weight_init=config["weight_init"], seed=seed))
    loss_fn = SoftmaxCrossEntropy()

    if config["optimizer"] == "Adam":
        optimizer = Adam(lr=config["learning_rate"])
    elif config["optimizer"] == "SGD":
        optimizer = SGD(lr=config["learning_rate"])
    else:
        raise ValueError(f"Unknown optimizer type: {config['optimizer']}")

    net.compile(loss=loss_fn, optimizer=optimizer)
    return net


def train_model(net, x_train, y_train, x_val, y_val, config, seed=None):
    start_time = time.perf_counter()
    history = net.fit(
        x_train,
        y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        validation_data=(x_val, y_val),
        seed=seed,
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

    results = []

    for seed in SEEDS:
        np.random.seed(seed)
        print(f"\n--- Seed: {seed} ---")
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(seed=seed)

        model = build_model(seed, FIXED_CONFIG)
        history, training_time = train_model(
            model, x_train, y_train, x_val, y_val, FIXED_CONFIG, seed=seed
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
            "seed": seed,
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

    val_accs = [r["best_validation_accuracy"] for r in results]
    test_accs = [r["test_accuracy"] for r in results]
    mean_val = float(np.mean(val_accs))
    std_val = float(np.std(val_accs))
    mean_test = float(np.mean(test_accs))
    std_test = float(np.std(test_accs))
    best = max(results, key=lambda x: x["best_validation_accuracy"])

    metrics = {
        "experiment": "seed_stability",
        "fixed_configuration": {
            "architecture": FIXED_CONFIG["architecture"],
            "activation": FIXED_CONFIG["activation"],
            "dropout": FIXED_CONFIG["dropout"],
            "weight_init": FIXED_CONFIG["weight_init"],
            "optimizer": FIXED_CONFIG["optimizer"],
            "learning_rate": FIXED_CONFIG["learning_rate"],
            "batch_size": FIXED_CONFIG["batch_size"],
            "epochs": FIXED_CONFIG["epochs"],
        },
        "summary": {
            "mean_validation_accuracy": mean_val,
            "std_validation_accuracy": std_val,
            "mean_test_accuracy": mean_test,
            "std_test_accuracy": std_test,
            "best_seed": best["seed"],
        },
        "results": results,
    }

    metrics_path = save_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved experiment metrics to {metrics_path}")

    seed_labels = [str(r["seed"]) for r in results]
    val_accs_raw = [r["best_validation_accuracy"] for r in results]
    x = np.arange(len(seed_labels))

    plots_dir = save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(x, val_accs_raw, color="royalblue")
    plt.xticks(x, seed_labels)
    plt.xlabel("Seed")
    plt.ylabel("Best Validation Accuracy")
    plt.title("Seed vs Best Validation Accuracy")
    for i, v in enumerate(val_accs_raw):
        plt.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
    plt.grid(True, axis="y", alpha=0.3)
    acc_plot_path = plots_dir / "accuracy_vs_seed.png"
    plt.tight_layout()
    plt.savefig(acc_plot_path, dpi=300)
    plt.close()
    print(f"Saved plot to {acc_plot_path}")

    plt.figure(figsize=(8, 5))
    plt.errorbar(["Test Accuracy"], [mean_test], yerr=[std_test], fmt="o", capsize=8, capthick=2, color="green", markersize=10)
    plt.ylabel("Accuracy")
    plt.title("Mean Test Accuracy with Std Dev")
    plt.grid(True, axis="y", alpha=0.3)
    plt.ylim(max(0, mean_test - 3 * std_test - 0.01), min(1.0, mean_test + 3 * std_test + 0.01))
    error_plot_path = plots_dir / "test_accuracy_error_bar.png"
    plt.tight_layout()
    plt.savefig(error_plot_path, dpi=300)
    plt.close()
    print(f"Saved plot to {error_plot_path}")

    print(f"\n{'=' * 30}")
    print(f"Mean Val Accuracy : {mean_val * 100:.2f}% ± {std_val * 100:.2f}%")
    print(f"Mean Test Accuracy: {mean_test * 100:.2f}% ± {std_test * 100:.2f}%")
    print(f"Best Seed         : {best['seed']}")
    print(f"{'=' * 30}")


if __name__ == "__main__":
    main()
