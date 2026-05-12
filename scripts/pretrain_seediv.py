#!/usr/bin/env python3
"""Pretrain the EEG-Conformer backbone on SEED-IV and save encoder weights."""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [%(levelname)s]  %(message)s")
logger = logging.getLogger("pretrain_seediv")

SEED_IV_N_CLASSES = 4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seediv_root", type=str, default="data/seed_iv")
    p.add_argument("--output_ckpt", type=str,
                   default="checkpoints/eeg_conformer_seediv.pt")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_seediv(root: Path, target_fs: int = 100) -> tuple[np.ndarray, np.ndarray]:
    import scipy.io as sio
    from data.eeg_features import extract_features

    all_epochs, all_labels = [], []
    mat_files = sorted(root.glob("**/*.mat"))[:15]
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under {root}")

    for mat_path in mat_files:
        try:
            mat = sio.loadmat(str(mat_path))
        except Exception as exc:
            logger.warning("Skipping %s: %s", mat_path, exc)
            continue

        for key, value in mat.items():
            if key.startswith("_") or not isinstance(value, np.ndarray):
                continue
            if value.ndim != 3 or value.shape[0] not in (30, 62):
                continue

            channels, samples, trials = value.shape
            use_channels = min(channels, 30)
            for trial_idx in range(trials):
                epoch_data = value[:use_channels, :, trial_idx]
                if samples > 1000:
                    from scipy.signal import resample_poly
                    epoch_data = resample_poly(epoch_data, 1, 2, axis=-1)

                for start in range(0, epoch_data.shape[1] - 500 + 1, 500):
                    segment = epoch_data[:, start:start + 500]
                    if segment.shape != (use_channels, 500):
                        continue
                    if use_channels < 30:
                        pad = np.zeros((30 - use_channels, 500))
                        segment = np.concatenate([segment, pad], axis=0)
                    all_epochs.append(segment[np.newaxis])
                    all_labels.append(trial_idx % SEED_IV_N_CLASSES)

    if not all_epochs:
        raise RuntimeError("No usable SEED-IV EEG segments found.")

    epochs = np.concatenate(all_epochs, axis=0).astype(np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    logger.info("SEED-IV: %d epochs loaded from %d files.", len(epochs), len(mat_files))

    features = extract_features(epochs, fs=float(target_fs))
    return features, labels


def build_conformer(n_classes: int = SEED_IV_N_CLASSES):
    from models.eeg_conformer import EEGConformer

    encoder = EEGConformer(
        n_channels=30,
        n_features=10,
        temporal_kernel=25,
        n_temp_filters=40,
        depth_mult=2,
        pool_size=3,
        conformer_heads=8,
        conformer_ff_dim=512,
        conformer_layers=2,
        embed_dim=256,
        dropout=0.5,
    )

    class ConformerWithHead(nn.Module):
        def __init__(self, eeg_encoder, n_out):
            super().__init__()
            self.eeg_encoder = eeg_encoder
            self.head = nn.Linear(256, n_out)

        def forward(self, x):
            return self.head(self.eeg_encoder(x))

    return ConformerWithHead(encoder, n_classes)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; using CPU.")
        device = "cpu"

    root = Path(args.seediv_root)
    if not root.exists():
        raise FileNotFoundError(f"SEED-IV root not found: {root}")

    X, y = load_seediv(root)
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    n_val = max(1, int(0.15 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=0)

    model = build_conformer().to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(1) == yb).sum().item()
                total += len(yb)
        acc = correct / max(total, 1)
        if epoch % 10 == 0:
            logger.info("Epoch %d/%d  val_acc=%.4f", epoch, args.epochs, acc)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    out = Path(args.output_ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state or model.state_dict(), str(out))
    logger.info("Saved SEED-IV checkpoint -> %s  (best val acc=%.4f)", out, best_acc)


if __name__ == "__main__":
    main()
