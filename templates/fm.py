"""Official-kit Factorization Machine. Do not change the math in draft 0."""

from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def _adam(self, gV, gW):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in (
            (self.V, gV, self.mV, self.vV),
            (self.W, gW, self.mW, self.vW),
        ):
            M *= b1
            M += (1 - b1) * G
            Vv *= b2
            Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)

    def step_logloss(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V
        gW += self.l2 * self.W
        self._adam(gV, gW)
        self.b -= self.lr * g.sum()
        p = sigmoid(z)
        return float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))

    def step_bpr(self, X, y):
        z, E, S = self.logits(X)
        pos = np.where(y > 0.5)[0]
        neg = np.where(y <= 0.5)[0]
        if len(pos) == 0 or len(neg) == 0:
            return self.step_logloss(X, y)
        n = min(len(pos), len(neg), len(y))
        pi = pos[np.arange(n) % len(pos)]
        ni = neg[np.arange(n) % len(neg)]
        diff = z[pi] - z[ni]
        sig = sigmoid(diff)
        loss = float(-np.mean(np.log(sig + 1e-9)))
        gpair = ((sig - 1.0) / n).astype(np.float32)
        g = np.zeros(len(y), dtype=np.float32)
        np.add.at(g, pi, gpair)
        np.add.at(g, ni, -gpair)
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V
        gW += self.l2 * self.W
        self._adam(gV, gW)
        self.b -= self.lr * g.sum()
        return loss

    def predict(self, X, bs=200_000):
        outs = [self.logits(X[i : i + bs])[0] for i in range(0, len(X), bs)]
        return np.concatenate(outs)
