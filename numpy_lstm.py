"""
A small, self-contained LSTM implemented from scratch in NumPy
(forward pass + full backprop-through-time + Adam optimizer).

Why not PyTorch/TensorFlow: this project runs in a constrained
sandbox where those frameworks pull in multi-gigabyte CUDA toolchains
that don't fit on disk, and the user asked to avoid heavier NN
frameworks anyway. A single-layer LSTM regressor is small enough to
implement directly and is fully sufficient for one-step-ahead
throughput forecasting on short windows.

This is intentionally a plain vanilla LSTM (no cuDNN-style fusions,
no batching tricks beyond simple mini-batches) so every line is
readable and auditable.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class NumpyLSTMRegressor:
    def __init__(self, n_features: int, hidden_size: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_features = n_features
        self.hidden_size = hidden_size
        z = n_features + hidden_size
        scale = 1.0 / np.sqrt(z)

        def w():
            return rng.uniform(-scale, scale, size=(z, hidden_size))

        self.Wf, self.Wi, self.Wc, self.Wo = w(), w(), w(), w()
        self.bf = np.zeros(hidden_size)
        self.bi = np.zeros(hidden_size)
        self.bc = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)
        self.Wy = rng.uniform(-scale, scale, size=(hidden_size, 1))
        self.by = np.zeros(1)

        self._params = ["Wf", "Wi", "Wc", "Wo", "bf", "bi", "bc", "bo", "Wy", "by"]
        self._m = {p: np.zeros_like(getattr(self, p)) for p in self._params}
        self._v = {p: np.zeros_like(getattr(self, p)) for p in self._params}
        self._t = 0

    # ---------- forward ----------
    def _forward(self, X: np.ndarray):
        """X: (batch, seq_len, n_features). Returns y_hat and a cache
        for backprop.
        """
        batch, seq_len, _ = X.shape
        h = np.zeros((batch, self.hidden_size))
        c = np.zeros((batch, self.hidden_size))
        cache = []
        for t in range(seq_len):
            xt = X[:, t, :]
            z = np.concatenate([xt, h], axis=1)
            f = _sigmoid(z @ self.Wf + self.bf)
            i = _sigmoid(z @ self.Wi + self.bi)
            o = _sigmoid(z @ self.Wo + self.bo)
            g = np.tanh(z @ self.Wc + self.bc)
            c_new = f * c + i * g
            h_new = o * np.tanh(c_new)
            cache.append((z, f, i, o, g, c, c_new, h, h_new))
            c, h = c_new, h_new
        y_hat = h @ self.Wy + self.by
        return y_hat, cache

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_hat, _ = self._forward(X)
        return y_hat.reshape(-1)

    # ---------- backward ----------
    def _backward(self, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray, cache):
        batch = X.shape[0]
        grads = {p: np.zeros_like(getattr(self, p)) for p in self._params}

        dy = (y_hat - y.reshape(-1, 1)) / batch  # MSE grad
        h_last = cache[-1][8]
        grads["Wy"] = h_last.T @ dy
        grads["by"] = dy.sum(axis=0)

        dh_next = dy @ self.Wy.T
        dc_next = np.zeros((batch, self.hidden_size))

        for t in reversed(range(len(cache))):
            z, f, i, o, g, c_prev, c_new, h_prev, h_new = cache[t]

            dh = dh_next
            do = dh * np.tanh(c_new)
            do_raw = do * o * (1 - o)

            dc = dc_next + dh * o * (1 - np.tanh(c_new) ** 2)
            df = dc * c_prev
            df_raw = df * f * (1 - f)

            di = dc * g
            di_raw = di * i * (1 - i)

            dg = dc * i
            dg_raw = dg * (1 - g ** 2)

            grads["Wf"] += z.T @ df_raw
            grads["bf"] += df_raw.sum(axis=0)
            grads["Wi"] += z.T @ di_raw
            grads["bi"] += di_raw.sum(axis=0)
            grads["Wo"] += z.T @ do_raw
            grads["bo"] += do_raw.sum(axis=0)
            grads["Wc"] += z.T @ dg_raw
            grads["bc"] += dg_raw.sum(axis=0)

            dz = (
                df_raw @ self.Wf.T
                + di_raw @ self.Wi.T
                + do_raw @ self.Wo.T
                + dg_raw @ self.Wc.T
            )
            dh_next = dz[:, self.n_features :]
            dc_next = dc * f

        for p in grads:
            np.clip(grads[p], -5, 5, out=grads[p])
        return grads

    def _adam_step(self, grads, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        for p in self._params:
            g = grads[p]
            self._m[p] = beta1 * self._m[p] + (1 - beta1) * g
            self._v[p] = beta2 * self._v[p] + (1 - beta2) * (g ** 2)
            m_hat = self._m[p] / (1 - beta1 ** self._t)
            v_hat = self._v[p] / (1 - beta2 ** self._t)
            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, p, getattr(self, p) - update)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=15, batch_size=64, lr=0.01, verbose=True):
        n = len(X)
        rng = np.random.default_rng(0)
        for epoch in range(epochs):
            order = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                xb, yb = X[idx], y[idx]
                y_hat, cache = self._forward(xb)
                loss = np.mean((y_hat.reshape(-1) - yb) ** 2)
                epoch_loss += loss * len(idx)
                grads = self._backward(xb, yb, y_hat, cache)
                self._adam_step(grads, lr=lr)
            if verbose:
                print(f"  [numpy_lstm] epoch {epoch + 1}/{epochs}  mse={epoch_loss / n:.4f}")
        return self
