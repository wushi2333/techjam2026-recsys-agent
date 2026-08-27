"""Factorization Machine plus optional history pooling / DIN-lite."""

from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0, seq_len=0, seq_mode="none"):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2, self.k = lr, l2, k
        self.seq_len = int(seq_len or 0)
        self.seq_mode = seq_mode if self.seq_len > 0 else "none"
        self.Vseq = rng.normal(0, 0.01, k).astype(np.float32)
        self.bseq = np.float32(0.0)
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mVs = np.zeros_like(self.Vseq)
        self.vVs = np.zeros_like(self.Vseq)
        self.t = 0
        self.rng = rng

    def snapshot(self):
        return (
            self.V.copy(),
            self.W.copy(),
            np.float32(self.b),
            self.Vseq.copy(),
            np.float32(self.bseq),
        )

    def restore(self, state):
        self.V, self.W, self.b, self.Vseq, self.bseq = state

    def fm_logits(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def _seq_ctx(self, X, H, M):
        HE = self.V[H]
        target = self.V[X[:, 1]]
        if self.seq_mode == "din":
            scores = (HE * target[:, None, :]).sum(-1)
            scores = scores + (1.0 - M) * (-1e9)
            scores = scores - scores.max(axis=1, keepdims=True)
            ex = np.exp(np.clip(scores, -30, 30)) * M
            attn = ex / (ex.sum(axis=1, keepdims=True) + 1e-9)
            ctx = (attn[:, :, None] * HE).sum(1)
            return ctx, HE, attn
        w = M[:, :, None]
        ctx = (HE * w).sum(1) / (w.sum(1) + 1e-9)
        return ctx, HE, None

    def logits(self, X, H=None, M=None):
        z, E, S = self.fm_logits(X)
        if self.seq_len <= 0 or H is None:
            return z, E, S, None
        ctx, HE, attn = self._seq_ctx(X, H, M)
        z = z + (ctx * self.Vseq).sum(1) + self.bseq
        return z, E, S, (ctx, HE, attn, M)

    def _adam_pair(self, P, G, M, Vv):
        M *= 0.9
        M += 0.1 * G
        Vv *= 0.999
        Vv += 0.001 * (G * G)
        P -= self.lr * (M / (1 - 0.9 ** self.t)) / (np.sqrt(Vv / (1 - 0.999 ** self.t)) + 1e-8)

    def _apply_grads(self, X, E, S, g, extra, H=None):
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        if extra is not None:
            ctx, HE, attn, M = extra
            gVs = (g[:, None] * ctx).sum(0)
            dctx = g[:, None] * self.Vseq
            if attn is None:
                denom = M.sum(1, keepdims=True) + 1e-9
                dHE = (dctx[:, None, :] * M[:, :, None]) / denom[:, :, None]
            else:
                dHE = attn[:, :, None] * dctx[:, None, :]
            np.add.at(gV, H, dHE)
            gV += self.l2 * self.V
            gW += self.l2 * self.W
            self.t += 1
            self._adam_pair(self.V, gV, self.mV, self.vV)
            self._adam_pair(self.W, gW, self.mW, self.vW)
            self._adam_pair(self.Vseq, gVs + self.l2 * self.Vseq, self.mVs, self.vVs)
            self.b -= self.lr * g.sum()
            self.bseq -= self.lr * g.sum()
            return
        gV += self.l2 * self.V
        gW += self.l2 * self.W
        self.t += 1
        self._adam_pair(self.V, gV, self.mV, self.vV)
        self._adam_pair(self.W, gW, self.mW, self.vW)
        self.b -= self.lr * g.sum()

    def step_logloss(self, X, y, H=None, M=None, users=None):
        B = len(y)
        z, E, S, extra = self.logits(X, H, M)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        self._apply_grads(X, E, S, g, extra, H)
        p = sigmoid(z)
        return float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))

    def step_bpr(self, X, y, H=None, M=None, users=None):
        if users is None:
            return self.step_logloss(X, y, H, M)
        z, E, S, extra = self.logits(X, H, M)
        g = np.zeros(len(y), dtype=np.float32)
        loss = 0.0
        npairs = 0
        buckets: dict = {}
        for i, user in enumerate(users):
            buckets.setdefault(user, []).append(i)
        for idxs in buckets.values():
            pos = [idxs[j] for j, v in enumerate(y[idxs]) if v > 0.5]
            neg = [idxs[j] for j, v in enumerate(y[idxs]) if v <= 0.5]
            if not pos or not neg:
                continue
            n_samp = min(len(pos) * len(neg), 32)
            for _ in range(n_samp):
                p = pos[int(self.rng.integers(len(pos)))]
                n = neg[int(self.rng.integers(len(neg)))]
                s = sigmoid(z[p] - z[n])
                loss += float(-np.log(s + 1e-9))
                c = float(s - 1.0)
                g[p] += c
                g[n] -= c
                npairs += 1
        if npairs == 0:
            return self.step_logloss(X, y, H, M)
        g /= npairs
        self._apply_grads(X, E, S, g, extra, H)
        return loss / npairs

    def step_bpr_global(self, X, y, H=None, M=None, users=None):
        """Cross-user pairwise margin. Empirically stronger here; not within-user BPR."""
        z, E, S, extra = self.logits(X, H, M)
        pos = np.where(y > 0.5)[0]
        neg = np.where(y <= 0.5)[0]
        if len(pos) == 0 or len(neg) == 0:
            return self.step_logloss(X, y, H, M)
        n = min(len(pos), len(neg), len(y))
        pi = pos[np.arange(n) % len(pos)]
        ni = neg[np.arange(n) % len(neg)]
        sig = sigmoid(z[pi] - z[ni])
        gpair = ((sig - 1.0) / n).astype(np.float32)
        g = np.zeros(len(y), dtype=np.float32)
        np.add.at(g, pi, gpair)
        np.add.at(g, ni, -gpair)
        self._apply_grads(X, E, S, g, extra, H)
        return float(-np.mean(np.log(sig + 1e-9)))

    def step_listwise(self, X, y, H=None, M=None, users=None):
        z, E, S, extra = self.logits(X, H, M)
        g = np.zeros(len(y), dtype=np.float32)
        loss = 0.0
        n_groups = 0
        buckets = {}
        for i, user in enumerate(users):
            buckets.setdefault(user, []).append(i)
        for idxs in buckets.values():
            if len(idxs) < 2:
                continue
            yy = y[idxs]
            npos = float(yy.sum())
            if npos <= 0 or npos >= len(idxs):
                continue
            zz = z[idxs]
            zz = zz - zz.max()
            e = np.exp(np.clip(zz, -30, 30))
            p = e / e.sum()
            yn = yy / npos
            g[idxs] = (p - yn).astype(np.float32)
            loss += float(-(yn * np.log(p + 1e-9)).sum())
            n_groups += 1
        if n_groups == 0:
            return self.step_logloss(X, y, H, M)
        g /= n_groups
        self._apply_grads(X, E, S, g, extra, H)
        return loss / n_groups

    def predict(self, X, H=None, M=None, bs=200_000):
        outs = []
        for i in range(0, len(X), bs):
            sl = slice(i, i + bs)
            hh = None if H is None else H[sl]
            mm = None if M is None else M[sl]
            outs.append(self.logits(X[sl], hh, mm)[0])
        return np.concatenate(outs)
