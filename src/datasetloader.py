import os
import sys
import urllib.request
import numpy as np

URLS = [
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
    "https://s3.amazonaws.com/img-datasets/mnist.npz",
]
_CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(_CACHE_DIR, "mnist.npz")


def _report(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        sys.stdout.write(f"\rDownloading MNIST: {pct:5.1f}%")
        sys.stdout.flush()


def download():
    if os.path.exists(CACHE_PATH):
        return CACHE_PATH

    last_err = None
    for url in URLS:
        try:
            print(f"Fetching MNIST from {url}")
            urllib.request.urlretrieve(url, CACHE_PATH, _report)
            print("\nDone.")
            return CACHE_PATH
        except Exception as e:
            last_err = e

    raise RuntimeError(f"could not download MNIST: {last_err}")


def _one_hot(labels, num_classes=10):
    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def load_data(val_split=0.1, seed=42):
    path = download()
    with np.load(path) as data:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]

    x_train = (x_train.astype(np.float32).reshape(-1, 784) / 255.0)
    x_test = (x_test.astype(np.float32).reshape(-1, 784) / 255.0)
    y_train = _one_hot(y_train)
    y_test = _one_hot(y_test)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x_train))
    x_train, y_train = x_train[indices], y_train[indices]
    
    n_val = int(len(x_train) * val_split)
    x_val, y_val = x_train[:n_val], y_train[:n_val]
    x_train, y_train = x_train[n_val:], y_train[n_val:]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    
    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)