"""Optional DeepFM / DCNv2 heads on flattened field embeddings."""

from __future__ import annotations

import numpy as np

ARCHS = {"fm", "deepfm", "dcnv2"}
HID = 32


def relu(x):
    return np.maximum(x, 0.0)


class ArchHead:
    def __init__(self, arch: str, k: int, lr: float, seed: int) -> None:
        name = str(arch or "fm")
        self.arch = name if name in {"deepfm", "dcnv2"} else "none"
        self.k = int(k)
        self.lr = float(lr)
        self.rng = np.random.default_rng(seed)
        self.ready = False
        self.t = 0
        self.in_dim = 0
        self.W1 = self.b1 = self.W2 = self.b2 = None
        self.Wc = self.bc = self.Wo = self.bo = None
        self.m = {}
        self.v = {}

    def snapshot(self):
        if not self.ready:
            return {"arch": self.arch, "ready": False}
        out = {"arch": self.arch, "ready": True, "in_dim": self.in_dim, "t": self.t}
        for key in ("W1", "b1", "W2", "b2", "Wc", "bc", "Wo", "bo"):
            val = getattr(self, key)
            out[key] = None if val is None else val.copy()
        out["m"] = {k: a.copy() for k, a in self.m.items()}
        out["v"] = {k: a.copy() for k, a in self.v.items()}
        return out

    def restore(self, state) -> None:
        if not state or not state.get("ready"):
            return
        self.arch = state["arch"]
        self.ready = True
        self.in_dim = int(state["in_dim"])
        self.t = int(state.get("t") or 0)
        for key in ("W1", "b1", "W2", "b2", "Wc", "bc", "Wo", "bo"):
            val = state.get(key)
            setattr(self, key, None if val is None else val.copy())
        self.m = {k: a.copy() for k, a in (state.get("m") or {}).items()}
        self.v = {k: a.copy() for k, a in (state.get("v") or {}).items()}

    def _init_pair(self, name, shape):
        w = self.rng.normal(0, 0.01, shape).astype(np.float32)
        setattr(self, name, w)
        self.m[name] = np.zeros_like(w)
        self.v[name] = np.zeros_like(w)

    def _ensure(self, E) -> None:
        if self.arch == "none" or self.ready:
            return
        self.in_dim = int(E.shape[1] * E.shape[2])
        if self.arch == "deepfm":
            self._init_pair("W1", (self.in_dim, HID))
            self._init_pair("b1", (HID,))
            self._init_pair("W2", (HID,))
            self._init_pair("b2", ())
        else:
            self._init_pair("Wc", (self.in_dim, self.in_dim))
            self._init_pair("bc", (self.in_dim,))
            self._init_pair("Wo", (self.in_dim,))
            self._init_pair("bo", ())
        self.ready = True

    def logit(self, E):
        if self.arch == "none":
            return np.zeros(len(E), dtype=np.float32)
        self._ensure(E)
        x = E.reshape(len(E), -1)
        if self.arch == "deepfm":
            h = relu(x @ self.W1 + self.b1)
            return (h @ self.W2 + self.b2).astype(np.float32)
        wx = x @ self.Wc + self.bc
        x1 = x * wx + x
        return (x1 @ self.Wo + self.bo).astype(np.float32)

    def _adam(self, name, g) -> None:
        p = getattr(self, name)
        m = self.m[name]
        v = self.v[name]
        m *= 0.9
        m += 0.1 * g
        v *= 0.999
        v += 0.001 * (g * g)
        t = self.t
        p -= self.lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    def backward(self, g, E, X, gV) -> None:
        if self.arch == "none":
            return
        self._ensure(E)
        self.t += 1
        x = E.reshape(len(E), -1)
        g = np.asarray(g, dtype=np.float32)
        if self.arch == "deepfm":
            dx = self._bwd_deep(g, x)
        else:
            dx = self._bwd_cross(g, x)
        np.add.at(gV, X, dx.reshape(E.shape))

    def _bwd_deep(self, g, x):
        w1, w2 = self.W1, self.W2
        h_pre = x @ w1 + self.b1
        h = relu(h_pre)
        dh = g[:, None] * w2
        dh_pre = dh * (h_pre > 0)
        dx = dh_pre @ w1.T
        self._adam("W2", h.T @ g)
        self._adam("b2", np.float32(g.sum()))
        self._adam("W1", x.T @ dh_pre)
        self._adam("b1", dh_pre.sum(0))
        return dx

    def _bwd_cross(self, g, x):
        wc, wo = self.Wc, self.Wo
        wx = x @ wc + self.bc
        x1 = x * wx + x
        dx1 = g[:, None] * wo
        dwx = dx1 * x
        dx = dx1 * wx + dx1 + dwx @ wc.T
        self._adam("Wo", x1.T @ g)
        self._adam("bo", np.float32(g.sum()))
        self._adam("Wc", x.T @ dwx)
        self._adam("bc", dwx.sum(0))
        return dx
