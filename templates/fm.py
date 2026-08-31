"""Factorization Machine plus optional history pooling / DIN-lite."""

from __future__ import annotations

import math

import numpy as np

from archhead import ArchHead

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def play_pos_weights(y, play) -> np.ndarray:
    """WLR: positives ~ log1p(play_time), negatives stay 1. Main loss only."""
    yv = np.asarray(y, dtype=np.float64)
    pv = np.asarray(play, dtype=np.float64)
    w = np.ones(len(yv), dtype=np.float32)
    pos = yv > 0.5
    if not np.any(pos):
        return w
    raw = np.log1p(np.maximum(pv[pos], 0.0))
    med = float(np.median(raw))
    scale = med if med > 1e-6 else 1.0
    w[pos] = np.clip(raw / scale, 0.25, 8.0).astype(np.float32)
    return w


class FM:
    def __init__(
        self,
        dim,
        k=16,
        lr=0.001,
        l2=1e-6,
        seed=0,
        seq_len=0,
        seq_mode="none",
        arch="fm",
        bpr_pairs_cap=32,
        listwise_gain="uniform",
    ):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2, self.k = lr, l2, k
        self.seq_len = int(seq_len or 0)
        self.seq_mode = seq_mode if self.seq_len > 0 else "none"
        self.Vseq = rng.normal(0, 0.01, k).astype(np.float32)
        self.bseq = np.float32(0.0)
        self.b_click = np.float32(0.0)
        self.b_cwm = np.float32(0.0)
        self.W_cwm = rng.normal(0, 0.01, k).astype(np.float32)
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mVs = np.zeros_like(self.Vseq)
        self.vVs = np.zeros_like(self.Vseq)
        self.mWcwm = np.zeros_like(self.W_cwm)
        self.vWcwm = np.zeros_like(self.W_cwm)
        self._cwm_gV = None
        self._cwm_gW = None
        self.t = 0
        self.rng = rng
        self.arch = ArchHead(arch, k, lr, seed)
        self.bpr_pairs_cap = max(1, int(bpr_pairs_cap or 32))
        self.listwise_gain = str(listwise_gain or "uniform")

    def snapshot(self):
        return (
            self.V.copy(),
            self.W.copy(),
            np.float32(self.b),
            self.Vseq.copy(),
            np.float32(self.bseq),
            np.float32(self.b_click),
            np.float32(self.b_cwm),
            self.W_cwm.copy(),
            self.arch.snapshot(),
        )

    def restore(self, state):
        if len(state) == 7:
            self.V, self.W, self.b, self.Vseq, self.bseq, self.b_click, self.b_cwm = state
            return
        self.V, self.W, self.b, self.Vseq, self.bseq, self.b_click, self.b_cwm, self.W_cwm, arch_s = state
        self.arch.restore(arch_s)

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
        extra = None
        if self.seq_len > 0 and H is not None:
            ctx, HE, attn = self._seq_ctx(X, H, M)
            z = z + (ctx * self.Vseq).sum(1) + self.bseq
            extra = (ctx, HE, attn, M)
        z = z + self.arch.logit(E)
        return z, E, S, extra

    def _adam_pair(self, P, G, M, Vv):
        M *= 0.9
        M += 0.1 * G
        Vv *= 0.999
        Vv += 0.001 * (G * G)
        P -= self.lr * (M / (1 - 0.9 ** self.t)) / (np.sqrt(Vv / (1 - 0.999 ** self.t)) + 1e-8)

    def _apply_grads(self, X, E, S, g, extra, H=None, g_cwm=None):
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        if g_cwm is not None:
            np.add.at(gW, X, g_cwm[:, None])
            np.add.at(gV, X, g_cwm[:, None, None] * (S[:, None, :] - E))
        gVs = None
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
        if self._cwm_gV is not None:
            gV += self._cwm_gV
            self._cwm_gV = None
        self.arch.backward(g, E, X, gV)
        gV += self.l2 * self.V
        gW += self.l2 * self.W
        self.t += 1
        if self._cwm_gW is not None:
            self._adam_pair(self.W_cwm, self._cwm_gW, self.mWcwm, self.vWcwm)
            self._cwm_gW = None
        self._adam_pair(self.V, gV, self.mV, self.vV)
        self._adam_pair(self.W, gW, self.mW, self.vW)
        if gVs is not None:
            self._adam_pair(self.Vseq, gVs + self.l2 * self.Vseq, self.mVs, self.vVs)
            self.bseq -= self.lr * g.sum()
        self.b -= self.lr * g.sum()

    def _wlr_vec(self, y, aux):
        if not aux or not aux.get("wlr"):
            return None
        play = aux.get("play")
        if play is None:
            return None
        return play_pos_weights(y, play)

    def _mix_aux(self, z, g, aux, E=None, X=None):
        if not aux:
            return 0.0, None
        extra = 0.0
        click = aux.get("click")
        w_click = float(aux.get("w_click") or 0.0)
        if click is not None and w_click:
            p = sigmoid(z + self.b_click)
            extra += w_click * float(
                -np.mean(click * np.log(p + 1e-9) + (1 - click) * np.log(1 - p + 1e-9))
            )
            g += (w_click * (p - click) / max(len(z), 1)).astype(np.float32)
            self.b_click -= np.float32(self.lr * w_click * float((p - click).mean()))
        cwm_l, g_cwm = self._cwm_aux(z, g, aux, E=E, X=X)
        return extra + cwm_l, g_cwm

    def _cwm_aux(self, z, g, aux, E=None, X=None):
        play = aux.get("play")
        dur = aux.get("dur")
        w = float(aux.get("w_cwm") or 0.0)
        if play is None or dur is None or not w:
            return 0.0, None
        head = str(aux.get("cwm_head") or "independent")
        if head == "independent":
            if E is None:
                return 0.0, None
            pooled = E.mean(axis=1)
            pred = pooled @ self.W_cwm + self.b_cwm
        else:
            pooled = None
            pred = z + self.b_cwm
        y_log = np.log(np.maximum(play, 0.0) + 1.0)
        c_log = np.log(np.maximum(dur, 1.0) + 1.0)
        cens = play >= 0.95 * np.maximum(dur, 1.0)
        resid = pred - y_log
        t = (c_log - pred)
        sf = np.clip(0.5 * (1.0 - np.vectorize(math.erf)(np.clip(t / _SQRT2, -8, 8))), 1e-6, 1.0)
        phi = np.exp(-0.5 * np.clip(t, -20, 20) ** 2) / _SQRT2PI
        loss = np.where(cens, -np.log(sf), 0.5 * resid * resid)
        g_pred = np.where(cens, -phi / sf, resid).astype(np.float32)
        g_pred = (w * g_pred / max(len(z), 1)).astype(np.float32)
        self.b_cwm -= np.float32(self.lr * float(g_pred.sum()))
        if head == "shared":
            g += g_pred
            return w * float(loss.mean()), None
        n_fields = E.shape[1]
        d_pooled = g_pred[:, None] * self.W_cwm
        dE = d_pooled[:, None, :] / n_fields
        gV = np.zeros_like(self.V)
        if X is not None:
            np.add.at(gV, X, dE)
        self._cwm_gV = gV
        self._cwm_gW = (pooled * g_pred[:, None]).sum(0) + self.l2 * self.W_cwm
        return w * float(loss.mean()), None

    def step_logloss(self, X, y, H=None, M=None, users=None, aux=None):
        B = len(y)
        z, E, S, extra = self.logits(X, H, M)
        p = sigmoid(z)
        bce = -(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
        w = self._wlr_vec(y, aux)
        if w is None:
            g = ((p - y) / B).astype(np.float32)
            loss = float(np.mean(bce))
        else:
            mass = float(np.sum(w)) + 1e-6
            g = ((p - y) * w / mass).astype(np.float32)
            loss = float(np.sum(bce * w) / mass)
        aux_l, g_cwm = self._mix_aux(z, g, aux, E, X)
        self._apply_grads(X, E, S, g, extra, H, g_cwm)
        return loss + aux_l

    def step_bpr(self, X, y, H=None, M=None, users=None, aux=None):
        if users is None:
            return self.step_logloss(X, y, H, M)
        z, E, S, extra = self.logits(X, H, M)
        g = np.zeros(len(y), dtype=np.float32)
        loss = 0.0
        npairs = 0.0
        w = self._wlr_vec(y, aux)
        buckets: dict = {}
        for i, user in enumerate(users):
            buckets.setdefault(user, []).append(i)
        for idxs in buckets.values():
            pos = [idxs[j] for j, v in enumerate(y[idxs]) if v > 0.5]
            neg = [idxs[j] for j, v in enumerate(y[idxs]) if v <= 0.5]
            if not pos or not neg:
                continue
            n_samp = min(len(pos) * len(neg), self.bpr_pairs_cap)
            for _ in range(n_samp):
                p = pos[int(self.rng.integers(len(pos)))]
                n = neg[int(self.rng.integers(len(neg)))]
                s = sigmoid(z[p] - z[n])
                wp = float(w[p]) if w is not None else 1.0
                loss += float(-np.log(s + 1e-9)) * wp
                c = float(s - 1.0) * wp
                g[p] += c
                g[n] -= c
                npairs += wp
        if npairs <= 0:
            return self.step_logloss(X, y, H, M, aux=aux)
        g /= npairs
        aux_l, g_cwm = self._mix_aux(z, g, aux, E, X)
        self._apply_grads(X, E, S, g, extra, H, g_cwm)
        return loss / npairs + aux_l

    def step_bpr_global(self, X, y, H=None, M=None, users=None, aux=None):
        """Cross-user pairwise margin. Empirically stronger here; not within-user BPR."""
        z, E, S, extra = self.logits(X, H, M)
        pos = np.where(y > 0.5)[0]
        neg = np.where(y <= 0.5)[0]
        if len(pos) == 0 or len(neg) == 0:
            return self.step_logloss(X, y, H, M, aux=aux)
        n = min(len(pos), len(neg), len(y))
        pi = pos[np.arange(n) % len(pos)]
        ni = neg[np.arange(n) % len(neg)]
        sig = sigmoid(z[pi] - z[ni])
        w = self._wlr_vec(y, aux)
        wp = w[pi] if w is not None else np.ones(n, dtype=np.float32)
        mass = float(np.sum(wp)) + 1e-6
        gpair = ((sig - 1.0) * wp / mass).astype(np.float32)
        g = np.zeros(len(y), dtype=np.float32)
        np.add.at(g, pi, gpair)
        np.add.at(g, ni, -gpair)
        aux_l, g_cwm = self._mix_aux(z, g, aux, E, X)
        self._apply_grads(X, E, S, g, extra, H, g_cwm)
        return float(-np.sum(wp * np.log(sig + 1e-9)) / mass) + aux_l

    def step_listwise(self, X, y, H=None, M=None, users=None, aux=None):
        if users is None:
            return self.step_logloss(X, y, H, M, aux=aux)
        if self.listwise_gain == "ndcg":
            return self._step_listwise_ndcg(X, y, H, M, users, aux)
        return self._step_listwise_uniform(X, y, H, M, users, aux)

    def _user_buckets(self, users):
        buckets: dict = {}
        for i, user in enumerate(users):
            buckets.setdefault(user, []).append(i)
        return buckets

    def _step_listwise_uniform(self, X, y, H, M, users, aux):
        z, E, S, extra = self.logits(X, H, M)
        g = np.zeros(len(y), dtype=np.float32)
        loss = 0.0
        n_groups = 0
        wlr = self._wlr_vec(y, aux)
        for idxs in self._user_buckets(users).values():
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
            if wlr is None:
                yn = yy / npos
            else:
                yn = yy * np.asarray(wlr[idxs], dtype=np.float64)
                mass = float(yn.sum())
                if mass <= 0:
                    continue
                yn = yn / mass
            g[idxs] = (p - yn).astype(np.float32)
            loss += float(-(yn * np.log(p + 1e-9)).sum())
            n_groups += 1
        if n_groups == 0:
            return self.step_logloss(X, y, H, M, aux=aux)
        g /= n_groups
        aux_l, g_cwm = self._mix_aux(z, g, aux, E, X)
        self._apply_grads(X, E, S, g, extra, H, g_cwm)
        return loss / n_groups + aux_l

    def _step_listwise_ndcg(self, X, y, H, M, users, aux):
        z, E, S, extra = self.logits(X, H, M)
        g = np.zeros(len(y), dtype=np.float32)
        loss = 0.0
        mass = 0.0
        cap = self.bpr_pairs_cap
        wlr = self._wlr_vec(y, aux)
        for idxs in self._user_buckets(users).values():
            if len(idxs) < 2:
                continue
            yy = y[idxs]
            npos = int(yy.sum())
            if npos <= 0 or npos >= len(idxs):
                continue
            zz = z[idxs]
            n = len(idxs)
            order = np.argsort(-zz, kind="mergesort")
            ranks = np.empty(n, dtype=np.int32)
            ranks[order] = np.arange(n, dtype=np.int32)
            disc = np.zeros(n, dtype=np.float64)
            top = min(n, 5)
            disc[:top] = 1.0 / np.log2(np.arange(top, dtype=np.float64) + 2.0)
            item_d = disc[ranks]
            idcg = float(disc[: min(npos, 5)].sum()) or 1e-9
            pos = [j for j, v in enumerate(yy) if v > 0.5]
            neg = [j for j, v in enumerate(yy) if v <= 0.5]
            n_samp = min(len(pos) * len(neg), cap)
            for _ in range(n_samp):
                p = pos[int(self.rng.integers(len(pos)))]
                nidx = neg[int(self.rng.integers(len(neg)))]
                delta = abs(item_d[p] - item_d[nidx]) / idcg
                if delta <= 0:
                    continue
                s = sigmoid(zz[p] - zz[nidx])
                wp = float(wlr[idxs[p]]) if wlr is not None else 1.0
                delta_w = delta * wp
                w = delta_w * float(s - 1.0)
                g[idxs[p]] += w
                g[idxs[nidx]] -= w
                loss += float(-delta_w * np.log(s + 1e-9))
                mass += delta_w
        if mass <= 0:
            return self.step_logloss(X, y, H, M, aux=aux)
        g /= mass
        aux_l, g_cwm = self._mix_aux(z, g, aux, E, X)
        self._apply_grads(X, E, S, g, extra, H, g_cwm)
        return loss / mass + aux_l

    def predict(self, X, H=None, M=None, bs=200_000):
        outs = []
        for i in range(0, len(X), bs):
            sl = slice(i, i + bs)
            hh = None if H is None else H[sl]
            mm = None if M is None else M[sl]
            outs.append(self.logits(X[sl], hh, mm)[0])
        return np.concatenate(outs)
