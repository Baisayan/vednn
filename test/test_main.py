import numpy as np
import pytest

from nn import Network, Dense, Dropout, ReLU, Sigmoid, SoftmaxCrossEntropy, MSE, Adam, SGD

EPS = 1e-5
ATOL = 1e-6

def numeric_grad(scalar_fn, x):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + EPS
        plus = scalar_fn()
        x[idx] = orig - EPS
        minus = scalar_fn()
        x[idx] = orig
        grad[idx] = (plus - minus) / (2 * EPS)
        it.iternext()
    return grad


def one_hot(idx, n_classes):
    out = np.zeros((idx.shape[0], n_classes))
    out[np.arange(idx.shape[0]), idx] = 1.0
    return out


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# Dense layer
def test_dense_forward_shape_and_value(rng):
    d = Dense(4, 3, seed=0)
    x = rng.normal(size=(5, 4))
    out = d.forward(x)
    assert out.shape == (5, 3)
    np.testing.assert_allclose(out, x @ d.W + d.b)

def test_dense_gradients(rng):
    d = Dense(4, 3, seed=0)
    x = rng.normal(size=(6, 4))
    y = one_hot(rng.integers(0, 3, size=6), 3)
    loss_fn = SoftmaxCrossEntropy()

    def loss():
        return loss_fn.forward(d.forward(x), y)

    loss()
    grad_in = d.backward(loss_fn.backward(d.output, y))
    np.testing.assert_allclose(numeric_grad(loss, d.W), d.dW, atol=ATOL)
    np.testing.assert_allclose(numeric_grad(loss, d.b), d.db, atol=ATOL)
    np.testing.assert_allclose(numeric_grad(loss, x), grad_in, atol=ATOL)

def test_dense_init_schemes():
    he = Dense(100, 100, weight_init="he", seed=0)
    xa = Dense(100, 100, weight_init="xavier", seed=0)
    assert he.W.var() > xa.W.var()
    with pytest.raises(ValueError):
        Dense(4, 3, weight_init="bogus")


# Activations
def test_relu_forward():
    relu = ReLU()
    x = np.array([[-2.0, 0.0, 3.0],
                  [1.5, -4.0, 0.2]])
    expected = np.array([[0.0, 0.0, 3.0],
                         [1.5, 0.0, 0.2]])
    np.testing.assert_array_equal(relu.forward(x), expected)
    
def test_relu_gradient(rng):
    act = ReLU()
    x = rng.normal(size=(5, 4))
    x = np.sign(x) * (np.abs(x) + 0.5)
    g = rng.normal(size=(5, 4))

    def scalar():
        return np.sum(g * act.forward(x))

    scalar()
    np.testing.assert_allclose(numeric_grad(scalar, x), act.backward(g), atol=ATOL)

def test_sigmoid_forward():
    sig = Sigmoid()
    x = np.array([[0.0, 1.0]])
    expected = np.array([[0.5, 1.0 / (1.0 + np.exp(-1.0))]])
    np.testing.assert_allclose(sig.forward(x), expected)

def test_sigmoid_gradient(rng):
    act = Sigmoid()
    x = rng.normal(size=(5, 4))
    g = rng.normal(size=(5, 4))

    def scalar():
        return np.sum(g * act.forward(x))

    scalar()
    np.testing.assert_allclose(numeric_grad(scalar, x), act.backward(g), atol=ATOL)

def test_sigmoid_handles_extremes():
    sig = Sigmoid()
    out = sig.forward(np.array([[-1000.0, 0.0, 1000.0]]))
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, [[0.0, 0.5, 1.0]], atol=1e-9)


# Losses
def test_mse_value_and_gradient(rng):
    mse = MSE()
    p = rng.normal(size=(4, 3))
    y = rng.normal(size=(4, 3))
    np.testing.assert_allclose(mse.forward(p, y), np.mean((p - y) ** 2))

    def loss():
        return mse.forward(p, y)

    np.testing.assert_allclose(numeric_grad(loss, p), mse.backward(p, y), atol=ATOL)


def test_softmaxcrossentropy_value_and_gradient(rng):
    sce = SoftmaxCrossEntropy()
    logits = rng.normal(size=(4, 3))
    y = one_hot(rng.integers(0, 3, size=4), 3)

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    expected_loss = -np.sum(y * np.log(np.clip(probs, 1e-12, 1.0))) / 4
    np.testing.assert_allclose(sce.forward(logits, y), expected_loss)

    def loss():
        return sce.forward(logits, y)

    np.testing.assert_allclose(numeric_grad(loss, logits), sce.backward(logits, y), atol=1e-5)

def test_softmaxcrossentropy_handles_large_logits():
    sce = SoftmaxCrossEntropy()
    logits = np.array([[1000.0, 1001.0, 1002.0]])
    y = np.array([[0.0, 0.0, 1.0]])
    loss = sce.forward(logits, y)
    grad = sce.backward(logits, y)

    assert np.isfinite(loss)
    assert np.all(np.isfinite(grad))


# Dropout
def test_dropout_is_identity_in_eval(rng):
    drop = Dropout(0.5, seed=0)
    x = rng.normal(size=(8, 6))
    out = drop.forward(x, training=False)
    np.testing.assert_array_equal(out, x)
    g = rng.normal(size=(8, 6))
    np.testing.assert_array_equal(drop.backward(g), g)


def test_dropout_rate_zero_is_identity(rng):
    drop = Dropout(0.0, seed=0)
    x = rng.normal(size=(8, 6))
    np.testing.assert_array_equal(drop.forward(x, training=True), x)

def test_dropout_train_and_eval_differ(rng):
    drop = Dropout(0.5, seed=0)
    x = np.ones((100, 20))
    train = drop.forward(x, training=True)
    test = drop.forward(x, training=False)

    assert not np.array_equal(train, test)

def test_dropout_mask_and_scaling(rng):
    rate = 0.4
    drop = Dropout(rate, seed=0)
    x = np.ones((1000, 1000))
    out = drop.forward(x, training=True)

    kept = out > 0
    np.testing.assert_allclose(out[kept], 1.0 / (1.0 - rate))
    assert abs(kept.mean() - (1.0 - rate)) < 0.01
    assert abs(out.mean() - 1.0) < 0.01


def test_dropout_backward_uses_mask(rng):
    drop = Dropout(0.5, seed=3)
    x = rng.normal(size=(8, 6))
    drop.forward(x, training=True)
    g = rng.normal(size=(8, 6))
    np.testing.assert_array_equal(drop.backward(g), g * drop.mask)


def test_dropout_invalid_rate():
    with pytest.raises(ValueError):
        Dropout(1.0)


# optimizers
def test_sgd_updates_parameters():
    layer = Dense(4, 3, seed=0)
    layer.dW = np.ones_like(layer.W)
    layer.db = np.ones_like(layer.b)
    old_W = layer.W.copy()
    old_b = layer.b.copy()
    SGD(lr=0.1).step([layer])

    assert not np.array_equal(old_W, layer.W)
    assert not np.array_equal(old_b, layer.b)

def test_sgd_momentum_creates_velocity():
    layer = Dense(4, 3, seed=0)
    layer.dW = np.ones_like(layer.W)
    layer.db = np.ones_like(layer.b)
    opt = SGD(lr=0.1, momentum=0.9)
    opt.step([layer])

    assert len(opt._velocity) > 0

def test_adam_updates_parameters():
    layer = Dense(4, 3, seed=0)
    layer.dW = np.ones_like(layer.W)
    layer.db = np.ones_like(layer.b)
    old_W = layer.W.copy()
    Adam(lr=1e-3).step([layer])

    assert not np.array_equal(old_W, layer.W)

def test_adam_internal_state():
    layer = Dense(4, 3, seed=0)
    layer.dW = np.ones_like(layer.W)
    layer.db = np.ones_like(layer.b)
    opt = Adam()
    opt.step([layer])
    opt.step([layer])

    assert opt.t == 2
    assert len(opt._m) > 0
    assert len(opt._v) > 0


# Whole-network gradient check
def test_full_network_param_gradients(rng):
    net = Network()
    net.add(Dense(6, 5, seed=0))
    net.add(ReLU())
    net.add(Dense(5, 3, seed=1))
    
    loss = SoftmaxCrossEntropy()
    x = rng.normal(size=(4, 6))
    y = one_hot(rng.integers(0, 3, size=4), 3)

    def total_loss():
        return loss.forward(net.forward(x, training=False), y)

    preds = net.forward(x, training=False)
    _ = loss.forward(preds, y)
    net.backward(loss.backward(preds, y))

    for layer in net.layers:
        for param, grad in layer.params_and_grads():
            np.testing.assert_allclose(
                numeric_grad(total_loss, param), grad, atol=ATOL,
            )


# End-to-end learning
@pytest.mark.parametrize("optimizer", [Adam(1e-2), SGD(0.5, momentum=0.9), SGD(0.5, momentum=0.0)])
def test_network_learns_synthetic(optimizer, rng):
    n, d, c = 400, 10, 3
    X = rng.normal(size=(n, d))
    W = rng.normal(size=(d, c))
    Y = one_hot(np.argmax(X @ W, axis=1), c)

    net = Network()
    net.add(Dense(d, 16, seed=0))
    net.add(ReLU())
    net.add(Dense(16, c, seed=1))
    net.compile(SoftmaxCrossEntropy(), optimizer)

    hist = net.fit(X, Y, epochs=40, batch_size=32, validation_data=(X, Y), seed=0)
    assert len(hist["loss"]) == 40
    assert len(hist["acc"]) == 40
    assert len(hist["val_loss"]) == 40
    assert len(hist["val_acc"]) == 40
    
    assert hist["loss"][-1] < hist["loss"][0]
    assert hist["acc"][-1] > 0.9
    test_loss, test_acc = net.evaluate(X, Y)
    assert test_loss < 0.5
    assert test_acc > 0.9

def test_network_evaluate(rng):
    X = rng.normal(size=(20, 4))
    Y = one_hot(rng.integers(0, 3, 20), 3)
    net = Network()
    net.add(Dense(4, 8, seed=0))
    net.add(ReLU())
    net.add(Dense(8, 3, seed=1))
    net.compile(SoftmaxCrossEntropy(), Adam())
    net.fit(X, Y, epochs=2, batch_size=4)
    loss, acc = net.evaluate(X, Y)

    assert np.isfinite(loss)
    assert 0.0 <= acc <= 1.0

def test_compile_required_before_evaluate(rng):
    net = Network()
    net.add(Dense(3, 2))
    with pytest.raises(RuntimeError):
        net.evaluate(rng.normal(size=(4, 3)), one_hot(rng.integers(0, 2, 4), 2))

def test_compile_required_before_fit(rng):
    net = Network()
    net.add(Dense(3, 2))
    with pytest.raises(RuntimeError):
        net.fit(rng.normal(size=(4, 3)), one_hot(rng.integers(0, 2, 4), 2), epochs=1, batch_size=2)