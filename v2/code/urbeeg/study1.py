"""Study 1: the original submission's design and architecture, re-run on real data.

Architecture (as in the original Methods and Supplementary S2):
  EEG tower      per-epoch DE + log-PSD features, 30 channels x 10; each channel is a token;
                 2-layer transformer encoder; mean pool; linear -> v_eeg (256)
                 (the original EEG-Conformer took the same 30 x 10 feature map; no SEED-IV
                 pretraining here, every network is trained from random initialisation)
  Env tower      graph over the grid cells (queen contiguity, inverse-squared-distance
                 weights, symmetric normalisation with self-loops); node features F;
                 the epoch's environmental vector is added to every node (H0 = F + 1 v^T);
                 two GCN layers (d -> 64 -> 128), a GRU over a sequence of length one,
                 mean pool over nodes, linear -> v_env (256)
  Fusion         bidirectional cross-modal attention, 8 heads, between v_eeg and v_env,
                 concatenated -> 512; 'uni' keeps only the EEG-query direction
  Head           512 -> 256 -> 5, ReLU, dropout 0.3
  Loss           cross-entropy + 0.1 * ||v_eeg - v_env||^2
Environmental vectors are drawn by contrast.EnvSampler (literature prior and controls).
"""
import numpy as np
import torch
import torch.nn as nn

from .contrast import EnvSampler, macro_f1


def grid_adjacency(grid) -> np.ndarray:
    """Queen contiguity with inverse-squared-distance weights, normalised D^-1/2 (A+I) D^-1/2."""
    r, c = grid["row"].to_numpy(), grid["col"].to_numpy()
    n = len(r)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dr, dc = abs(r[i] - r[j]), abs(c[i] - c[j])
            if i != j and dr <= 1 and dc <= 1:
                A[i, j] = 1.0 / (dr ** 2 + dc ** 2)
    A += np.eye(n)
    d = A.sum(1)
    return A / np.sqrt(np.outer(d, d))


class EEGTower(nn.Module):
    def __init__(self, n_ch=30, n_feat=10, d=64, out=256, drop=0.3, n_extra=0):
        super().__init__()
        self.embed = nn.Linear(n_feat, d)
        self.ch = nn.Parameter(torch.zeros(1, n_ch, d))
        self.extra = nn.Sequential(nn.Dropout(drop), nn.Linear(n_extra, d)) if n_extra else None
        self.extra_pos = nn.Parameter(torch.zeros(1, 1, d)) if n_extra else None
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, drop, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, 2)
        self.out = nn.Linear(d, out)

    def forward(self, x, t=None):                # x (B, 30, 10); t (B, n_extra) covariance token
        h = self.embed(x) + self.ch
        if self.extra is not None:
            h = torch.cat([h, self.extra(t)[:, None] + self.extra_pos], 1)
        return self.out(self.tf(h).mean(1))


class EnvTower(nn.Module):
    def __init__(self, A, F, out=256):
        super().__init__()
        self.register_buffer("A", torch.as_tensor(A, dtype=torch.float32))
        self.register_buffer("F", torch.as_tensor(F, dtype=torch.float32))
        d = F.shape[1]
        self.g1, self.g2 = nn.Linear(d, 64), nn.Linear(64, 128)
        self.gru = nn.GRU(128, 128, batch_first=True)
        self.out = nn.Linear(128, out)

    def forward(self, v):                        # (B, d)
        h = self.F[None] + v[:, None, :]         # broadcast the epoch vector to every node
        h = torch.relu(self.A @ self.g1(h))
        h = torch.relu(self.A @ self.g2(h))      # (B, N, 128)
        B, N, D = h.shape
        h, _ = self.gru(h.reshape(B * N, 1, D))  # sequence of length one
        return self.out(h.reshape(B, N, D).mean(1))


class Study1Model(nn.Module):
    def __init__(self, A, F, mode="bi", n_cls=5, drop=0.3, n_extra=0):
        super().__init__()
        self.mode = mode                          # 'bi', 'uni' or 'eeg_only'
        self.eeg = EEGTower(drop=drop, n_extra=n_extra)
        if mode != "eeg_only":
            self.env = EnvTower(A, F)
            self.att_e = nn.MultiheadAttention(256, 8, batch_first=True)
            if mode == "bi":
                self.att_v = nn.MultiheadAttention(256, 8, batch_first=True)
            self.proj = nn.Linear(512 if mode == "bi" else 256, 512)
        self.head = nn.Sequential(nn.Linear(512 if mode != "eeg_only" else 256, 256), nn.ReLU(),
                                  nn.Dropout(drop), nn.Linear(256, n_cls))

    def forward(self, x, v, t=None):
        ze = self.eeg(x, t)
        if self.mode == "eeg_only":
            return self.head(ze), torch.zeros((), device=x.device)
        zv = self.env(v)
        qe, qv = ze[:, None], zv[:, None]
        a_e, _ = self.att_e(qe, qv, qv)           # EEG queries environment
        fused = a_e[:, 0]
        if self.mode == "bi":
            a_v, _ = self.att_v(qv, qe, qe)       # environment queries EEG
            fused = torch.cat([fused, a_v[:, 0]], 1)
        return self.head(self.proj(fused)), ((ze - zv) ** 2).mean()


def train_eval(X, y, tr, va, te, cond, E, A, q, h, seed, mode="bi", lam=0.1,
               epochs=40, patience=6, batch=128, lr=1e-3, init_from=None, freeze_eeg=False,
               return_model=False, T=None, drop=0.3, wd=1e-4, return_proba=False):
    """Train on tr (environment per `cond`), early-stop on va (macro-F1), predict te."""
    torch.manual_seed(seed)
    s = EnvSampler(E, cond if mode != "eeg_only" else "noise", q=q, h=h, seed=seed)
    V = {k: (s.train(y[idx]) if k == "tr" else s.test(y[idx])) for k, idx in
         (("tr", tr), ("va", va), ("te", te))}
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
    model = Study1Model(A, E, mode=mode, drop=drop, n_extra=0 if T is None else T.shape[1])
    if init_from is not None:                     # environmental-domain shift: reuse EEG tower
        model.eeg.load_state_dict(init_from.eeg.state_dict())
    if freeze_eeg:
        for p in model.eeg.parameters():
            p.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                            weight_decay=wd)
    ce = nn.CrossEntropyLoss()
    Xtr, Vtr, Ytr = t(X[tr]), t(V["tr"]), torch.as_tensor(y[tr])
    Ttr = None if T is None else t(T[tr])
    tsel = lambda idx: None if T is None else t(T[idx])
    g = torch.Generator().manual_seed(seed)
    best, state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for b in torch.randperm(len(tr), generator=g).split(batch):
            logits, align = model(Xtr[b], Vtr[b], None if Ttr is None else Ttr[b])
            loss = ce(logits, Ytr[b]) + lam * align
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            f = macro_f1(y[va], model(t(X[va]), t(V["va"]), tsel(va))[0].argmax(1).numpy())
        if f > best + 1e-4:
            best, bad = f, 0
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(t(X[te]), t(V["te"]), tsel(te))[0], 1).numpy()
    pred = proba.argmax(1)
    if return_proba:
        return pred, proba, best
    return (pred, model) if return_model else pred
