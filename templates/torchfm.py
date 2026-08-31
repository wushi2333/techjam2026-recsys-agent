"""PyTorch FM. model_family=torch. Uses CUDA when present, else CPU."""

from __future__ import annotations

import time

import numpy as np


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("model_family=torch requires PyTorch") from exc
    return torch, nn, F


def pick_device(cfg, torch):
    want = str((cfg or {}).get("torch_device") or "auto")
    if want == "cpu":
        return torch.device("cpu")
    cuda = bool(torch.cuda.is_available())
    if want == "cuda":
        if not cuda:
            raise RuntimeError("torch_device=cuda but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if cuda else "cpu")


def _build_net(torch, nn, dim, k, n_fields, seq_len, seq_mode, arch):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.V = nn.Embedding(dim, k)
            self.W = nn.Embedding(dim, 1)
            nn.init.normal_(self.V.weight, 0.0, 0.01)
            nn.init.zeros_(self.W.weight)
            self.b = nn.Parameter(self.V.weight.new_zeros(1))
            self.seq_len = int(seq_len or 0)
            self.seq_mode = seq_mode if self.seq_len > 0 else "none"
            self.Vseq = nn.Parameter(self.V.weight.new_zeros(k))
            nn.init.normal_(self.Vseq, 0.0, 0.01)
            self.bseq = nn.Parameter(self.V.weight.new_zeros(1))
            self.arch = arch if arch in {"deepfm", "dcnv2"} else "fm"
            hid, in_dim = 32, n_fields * k
            self.mlp = None
            self.cross_w = self.cross_o = None
            if self.arch == "deepfm":
                self.mlp = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, 1))
            if self.arch == "dcnv2":
                self.cross_w = nn.Linear(in_dim, in_dim, bias=True)
                self.cross_o = nn.Linear(in_dim, 1, bias=True)

        def forward(self, X, H=None, M=None):
            E = self.V(X)
            S = E.sum(1)
            inter = 0.5 * ((S * S).sum(1) - (E * E).sum(dim=(1, 2)))
            z = self.b + self.W(X).sum(1).squeeze(-1) + inter
            if self.seq_len > 0 and H is not None:
                HE = self.V(H)
                if self.seq_mode == "din":
                    target = self.V(X[:, 1])
                    scores = (HE * target[:, None, :]).sum(-1)
                    scores = scores.masked_fill(M <= 0, -1e9)
                    attn = torch.softmax(scores, dim=1) * M
                    attn = attn / (attn.sum(1, keepdim=True) + 1e-9)
                    ctx = (attn[:, :, None] * HE).sum(1)
                else:
                    w = M[:, :, None]
                    ctx = (HE * w).sum(1) / (w.sum(1) + 1e-9)
                z = z + (ctx * self.Vseq).sum(1) + self.bseq
            x = E.reshape(E.size(0), -1)
            if self.mlp is not None:
                z = z + self.mlp(x).squeeze(-1)
            if self.cross_w is not None:
                xc = x * self.cross_w(x) + x
                z = z + self.cross_o(xc).squeeze(-1)
            return z

    return Net()


class TorchFM:
    def __init__(self, net, device) -> None:
        self.net = net
        self.device = device

    def predict(self, X, H=None, M=None, bs=8192):
        torch, _, _ = _import_torch()
        self.net.eval()
        out = []
        n = len(X)
        with torch.no_grad():
            for i in range(0, n, bs):
                sl = slice(i, i + bs)
                xt, ht, mt = _tensors(torch, self.device, X, H, M, sl)
                out.append(self.net(xt, ht, mt).detach().cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float64)

    def snapshot(self):
        return {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}

    def restore(self, state) -> None:
        self.net.load_state_dict(state)


def _tensors(torch, device, X, H, M, sl):
    xt = torch.as_tensor(np.asarray(X[sl]), dtype=torch.long, device=device)
    ht = mt = None
    if H is not None:
        ht = torch.as_tensor(np.asarray(H[sl]), dtype=torch.long, device=device)
        mt = torch.as_tensor(np.asarray(M[sl]), dtype=torch.float32, device=device)
    return xt, ht, mt


def _listwise(F, z, y, users, ndcg: bool, sw=None):
    umap: dict = {}
    for i, user in enumerate(users):
        umap.setdefault(user, []).append(i)
    loss = z.new_zeros(())
    n = 0
    for idxs in umap.values():
        if len(idxs) < 2:
            continue
        zz = z[idxs]
        yy = y[idxs]
        w = (2.0**yy - 1.0) if ndcg else yy
        if sw is not None:
            w = w * sw[idxs]
        if float(w.sum()) <= 0:
            continue
        logp = zz - zz.logsumexp(0)
        loss = loss + -(w * logp).sum() / (w.sum() + 1e-9)
        n += 1
    if n == 0:
        return F.binary_cross_entropy_with_logits(z, y)
    return loss / n


def _loss_of(F, torch, z, yt, users, loss_name, ndcg, w=None):
    if w is None:
        if loss_name == "bpr_global":
            perm = torch.randperm(len(z), device=z.device)
            return -F.logsigmoid(z - z[perm]).mean()
        if loss_name == "bpr":
            pos = yt > 0.5
            neg = ~pos
            if int(pos.sum()) == 0 or int(neg.sum()) == 0:
                return F.binary_cross_entropy_with_logits(z, yt)
            return -F.logsigmoid(z[pos].unsqueeze(1) - z[neg].unsqueeze(0)).mean()
        if loss_name == "listwise":
            return _listwise(F, z, yt, users, ndcg)
        return F.binary_cross_entropy_with_logits(z, yt)
    mass = w.sum() + 1e-6
    if loss_name == "bpr_global":
        perm = torch.randperm(len(z), device=z.device)
        return -(F.logsigmoid(z - z[perm]) * w).sum() / mass
    if loss_name == "bpr":
        pos = yt > 0.5
        neg = ~pos
        if int(pos.sum()) == 0 or int(neg.sum()) == 0:
            bce = F.binary_cross_entropy_with_logits(z, yt, reduction="none")
            return (bce * w).sum() / mass
        pair = -F.logsigmoid(z[pos].unsqueeze(1) - z[neg].unsqueeze(0))
        return (pair * w[pos].unsqueeze(1)).sum() / (w[pos].sum() * float(int(neg.sum())) + 1e-6)
    if loss_name == "listwise":
        return _listwise(F, z, yt, users, ndcg, sw=w)
    bce = F.binary_cross_entropy_with_logits(z, yt, reduction="none")
    return (bce * w).sum() / mass


def train_torch(enc, cfg, evaluate):
    torch, nn, F = _import_torch()
    from sampling import iter_user_batches
    from train import _eval_pack, _hist, _take, _user_mask, should_eval, train_limits

    Xtr, ytr, utr = enc["train"]
    Xva, yva, uva = enc["valid"]
    Htr, Mtr = _hist(enc, "train")
    Hva, Mva = _hist(enc, "valid")
    device = pick_device(cfg, torch)
    seed = int(cfg.get("seed") or 0)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    net = _build_net(
        torch,
        nn,
        int(enc["dim"]),
        int(cfg.get("k") or 16),
        int(Xtr.shape[1]),
        int(cfg.get("seq_len") or 0),
        str(cfg.get("seq_mode") or "none"),
        str(cfg.get("arch") or "fm"),
    ).to(device)
    opt = torch.optim.Adam(
        net.parameters(),
        lr=float(cfg.get("lr") or 0.001),
        weight_decay=float(cfg.get("l2") or 0.0),
    )
    model = TorchFM(net, device)
    rng = np.random.default_rng(seed)
    best, state, bad = -1.0, None, 0
    curves = []
    epochs, patience = train_limits(cfg)
    eval_every = int(cfg.get("eval_every") or 1)
    Xev, yev, uev, Hev, Mev = _eval_pack(enc)
    mask = _user_mask(uev, cfg.get("eval_user_frac") or 1.0, seed)
    cheap = (
        _take(Xev, mask),
        _take(yev, mask),
        _take(uev, mask),
        _take(Hev, mask),
        _take(Mev, mask),
    )
    bs = int(cfg["batch"])
    loss_name = str(cfg.get("loss") or "logloss")
    ndcg = str(cfg.get("listwise_gain") or "uniform") == "ndcg"
    play_all = None
    if cfg.get("wlr_play"):
        play_all = ((enc.get("aux") or {}).get("train") or {}).get("play")
    for ep in range(1, epochs + 1):
        t0 = time.time()
        net.train()
        losses = []
        if loss_name in ("bpr", "listwise"):
            slices = iter_user_batches(utr, bs, rng)
        else:
            perm = rng.permutation(len(ytr))
            slices = (perm[i : i + bs] for i in range(0, len(perm), bs))
        for sl in slices:
            xt, ht, mt = _tensors(torch, device, Xtr, Htr, Mtr, sl)
            yt = torch.as_tensor(np.asarray(ytr[sl]), dtype=torch.float32, device=device)
            users = [utr[int(j)] for j in sl]
            z = net(xt, ht, mt)
            wt = None
            if play_all is not None:
                from fm import play_pos_weights

                wt = torch.as_tensor(
                    play_pos_weights(ytr[sl], play_all[sl]),
                    dtype=torch.float32,
                    device=device,
                )
            loss = _loss_of(F, torch, z, yt, users, loss_name, ndcg, w=wt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses)) if losses else 0.0
        if not should_eval(ep, epochs, eval_every):
            print(
                f"  epoch {ep:2d} | loss {mean_loss:.4f} | (skip valid) | {time.time() - t0:.1f}s",
                flush=True,
            )
            continue
        cx, cy, cu, ch, cm = cheap
        va = evaluate(cu, cy, model.predict(cx, ch, cm))
        row = {
            "epoch": ep,
            "loss": mean_loss,
            "primary": float(va["primary"]),
            "GAUC": float(va["GAUC"]),
            "nDCG@5": float(va["nDCG@5"]),
            "sec": time.time() - t0,
        }
        curves.append(row)
        tag = "stop" if enc.get("stop") else ("cheap" if mask is not None else "valid")
        print(
            f"  epoch {ep:2d} | loss {row['loss']:.4f} | {tag} GAUC {row['GAUC']:.4f} "
            f"nDCG@5 {row['nDCG@5']:.4f} primary {row['primary']:.4f} | {row['sec']:.1f}s",
            flush=True,
        )
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            state = model.snapshot()
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop at epoch {ep}", flush=True)
                break
    if state is not None:
        model.restore(state)
    metrics = evaluate(uva, yva, model.predict(Xva, Hva, Mva))
    metrics = dict(metrics)
    metrics["torch_cuda"] = 1.0 if device.type == "cuda" else 0.0
    return model, metrics, curves
