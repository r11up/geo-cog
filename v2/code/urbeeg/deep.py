"""Deep models trained on single EEG epochs; a pair's prediction is the mean over its epochs.

Models
------
eegnet        braindecode EEGNet (Lawhern et al. 2018), EEG only.
conformer     braindecode EEGConformer (Song et al. 2023), EEG only.
dualtower     EEG tower (shallow conv -> time tokens -> transformer) and a scene tower with
              one token per scene category (greenery, buildings, sky, ground, vehicles,
              people, other), joined by bidirectional cross-attention: EEG time tokens
              attend to scene tokens and scene tokens attend to EEG time tokens.
              scene_input = 'real' | 'noise' | 'shuffled' gives the input controls.

Training: inner split 0 of the cell (people x scenes new to the model) is the validation
set for early stopping; the model is trained on the remaining training block. Targets are
the same z-scored targets as the linear models.
"""
import time

import numpy as np
import torch
import torch.nn as nn

from .dataset import EEG_DIR, epochs_per_scene
from .scene import FEATURES as COMPOSITION

UV_SCALE = 1e6 / 20.0          # volts -> roughly unit variance


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EpochStore:
    """All participants' epochs in memory (float16) with a pair -> epoch-rows index."""

    def __init__(self, participants):
        import pandas as pd

        xs, self.pair_epochs, offset = [], {}, 0
        for sub in participants:
            z = np.load(EEG_DIR / f"{sub}_epochs.npz")
            trials = pd.read_csv(EEG_DIR / f"{sub}_trials.csv")
            if "eeg_uv" in z:                      # compact export: float16 microvolts
                xs.append((z["eeg_uv"].astype(np.float32) / 20.0).astype(np.float16))
            else:
                xs.append((z["eeg"] * UV_SCALE).astype(np.float16))
            for stim, idx in epochs_per_scene(trials).items():
                self.pair_epochs[(sub, stim)] = idx + offset
            offset += len(trials)
        self.x = np.concatenate(xs)
        self.n_chans, self.n_times = self.x.shape[1:]

    def expand(self, data, rows):
        """Epoch indices and the pair row each epoch belongs to."""
        ep, owner = [], []
        for r in rows:
            e = self.pair_epochs[(data["participant"][r], data["stimulus"][r])]
            ep.append(e)
            owner.append(np.full(len(e), r))
        return np.concatenate(ep), np.concatenate(owner)


# ------------------------------------------------------------------ dual-tower model

class EEGTokens(nn.Module):
    """Shallow temporal + spatial convolution (EEG-Conformer style), average pooling into
    time tokens (25-sample window, stride 10: ~0.2 s windows every 80 ms at 125 Hz), then a
    small transformer over the tokens."""

    def __init__(self, n_chans, n_times, d=64, n_filters=40, layers=2, heads=4, drop=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, 25), padding=(0, 12)),
            nn.Conv2d(n_filters, n_filters, (n_chans, 1)),
            nn.BatchNorm2d(n_filters), nn.ELU(),
            nn.AvgPool2d((1, 25), (1, 10)), nn.Dropout(drop))
        self.proj = nn.Linear(n_filters, d)
        n_tokens = (n_times - 25) // 10 + 1
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d))
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers)

    def forward(self, x):                               # x: (B, C, T)
        h = self.conv(x.unsqueeze(1)).squeeze(2).transpose(1, 2)     # (B, T', F)
        return self.tf(self.proj(h) + self.pos)


class SceneTokens(nn.Module):
    """One token per category: value * learned direction + learned category embedding."""

    def __init__(self, n_cat, d=64):
        super().__init__()
        self.dir = nn.Parameter(torch.randn(n_cat, d) * 0.1)
        self.emb = nn.Parameter(torch.randn(n_cat, d) * 0.1)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU())

    def forward(self, s):                               # s: (B, n_cat), standardised
        return self.mlp(s.unsqueeze(-1) * self.dir + self.emb)


class DualTower(nn.Module):
    def __init__(self, n_chans, n_times, n_cat, n_out, d=64, heads=4, drop=0.3):
        super().__init__()
        self.eeg, self.scene = EEGTokens(n_chans, n_times, d), SceneTokens(n_cat, d)
        self.e2s = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.s2e = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ne, self.ns = nn.LayerNorm(d), nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, n_out))

    def forward(self, x, s, return_attention=False):
        e, c = self.eeg(x), self.scene(s)
        e2, a_es = self.e2s(e, c, c)      # EEG time tokens attend to scene categories
        c2, a_se = self.s2e(c, e, e)      # scene categories attend to EEG time tokens
        e, c = self.ne(e + e2), self.ns(c + c2)
        out = self.head(torch.cat([e.mean(1), c.mean(1)], dim=-1))
        return (out, a_es, a_se) if return_attention else out


# ------------------------------------------------------------------ training

def build(name, n_chans, n_times, n_out):
    from braindecode.models import EEGConformer, EEGNet

    if name == "eegnet":
        return EEGNet(n_chans=n_chans, n_outputs=n_out, n_times=n_times)
    if name == "conformer":
        return EEGConformer(n_outputs=n_out, n_chans=n_chans, n_times=n_times,
                            pool_time_length=38, pool_time_stride=8, num_layers=4, num_heads=8)
    if name.startswith("dualtower"):
        return DualTower(n_chans, n_times, len(COMPOSITION), n_out)
    raise ValueError(name)


def scene_matrix(data, rows, mode, seed, mu, sd):
    """Standardised scene composition per row; 'noise' and 'shuffled' are the controls."""
    scenes = np.array(sorted(data["scene"].index))
    if mode == "shuffled":
        perm = dict(zip(scenes, np.random.default_rng(seed).permutation(scenes)))
        stim = np.array([perm[s] for s in data["stimulus"][rows]])
    else:
        stim = data["stimulus"][rows]
    if mode == "noise":
        table = dict(zip(scenes, np.random.default_rng(seed).normal(size=(len(scenes), len(COMPOSITION)))))
        return np.array([table[s] for s in stim], np.float32)
    return ((data["scene"].loc[stim, COMPOSITION].to_numpy() - mu) / sd).astype(np.float32)


def train_predict(name, store, data, Y, sel_tr, sel_va, test, seed, scene_input="real",
                  max_epochs=40, patience=6, batch=128, lr=1e-3, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()
    n_out = Y.shape[1]
    model = build(name, store.n_chans, store.n_times, n_out).to(dev)
    uses_scene = name.startswith("dualtower")
    comp_tr = data["scene"].loc[data["stimulus"][sel_tr], COMPOSITION]
    mu, sd = comp_tr.mean().to_numpy(), comp_tr.std().to_numpy() + 1e-8

    def batches(rows, shuffle):
        ep, owner = store.expand(data, rows)
        order = np.random.permutation(len(ep)) if shuffle else np.arange(len(ep))
        S = scene_matrix(data, owner, scene_input, seed, mu, sd) if uses_scene else None
        for i in range(0, len(order), batch):
            j = order[i:i + batch]
            xb = torch.from_numpy(store.x[ep[j]].astype(np.float32)).to(dev)
            sb = torch.from_numpy(S[j]).to(dev) if uses_scene else None
            yield j, owner[j], xb, sb

    def forward(xb, sb):
        return model(xb, sb) if uses_scene else model(xb)

    def predict_pairs(rows):
        model.eval()
        sums, counts = {}, {}
        with torch.no_grad():
            for _, own, xb, sb in batches(rows, False):
                p = forward(xb, sb).float().cpu().numpy()
                for r, v in zip(own, p):
                    sums[r] = sums.get(r, 0) + v
                    counts[r] = counts.get(r, 0) + 1
        return np.stack([sums[r] / counts[r] for r in rows])

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.MSELoss()
    best, best_state, bad, t0 = np.inf, None, 0, time.time()
    for epoch in range(max_epochs):
        model.train()
        for _, own, xb, sb in batches(sel_tr, True):
            yb = torch.from_numpy(Y[own].astype(np.float32)).to(dev)
            opt.zero_grad()
            loss_fn(forward(xb, sb), yb).backward()
            opt.step()
        val = float(((predict_pairs(sel_va) - Y[sel_va]) ** 2).mean())
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    log(f"{name}/{scene_input}: {epoch + 1} epochs, best val MSE {best:.3f}, {time.time() - t0:.0f} s")
    return predict_pairs(test)
