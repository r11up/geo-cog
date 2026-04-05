"""Training loop with AdamW, cosine scheduling, and early stopping."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 15):
        self.patience = patience
        self.best     = -np.inf
        self.counter  = 0
        self.stop     = False

    def step(self, value: float) -> bool:
        if value > self.best + 1e-4:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


class Trainer:
    def __init__(self, model: nn.Module, cfg, device: str,
                 is_multimodal: bool = True,
                 loss_name: str = "compute_loss",
                 loss_kwargs: Optional[Dict[str, Any]] = None):
        self.model         = model.to(device)
        self.device        = device
        self.is_multimodal = is_multimodal
        self.loss_name     = loss_name
        self.loss_kwargs   = loss_kwargs or {}

        tc = cfg.train
        self.optimizer = AdamW(model.parameters(),
                               lr=tc.learning_rate,
                               weight_decay=tc.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer,
                                           T_max=tc.cosine_t_max,
                                           eta_min=1e-6)
        self.early_stopper  = EarlyStopping(patience=tc.patience)
        self.n_epochs       = tc.n_epochs
        self.checkpoint_dir = cfg.paths.checkpoint_dir

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              fold_id: int = 0) -> Dict:
        history    = {"train_loss": [], "train_acc": [],
                      "val_acc": [], "val_f1": [], "val_auc": []}
        best_val_f1 = -1.0
        best_ckpt   = self.checkpoint_dir / f"fold{fold_id}_best.pt"

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = self._train_epoch(train_loader)
            val_acc, val_f1, val_auc, _, _ = self.evaluate(val_loader)
            self.scheduler.step()

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)
            history["val_auc"].append(val_auc)

            logger.info(
                "Fold %d  Epoch %3d/%d  loss=%.4f  tr_acc=%.3f  "
                "val_acc=%.3f  val_f1=%.3f  (%.1fs)",
                fold_id, epoch, self.n_epochs,
                tr_loss, tr_acc, val_acc, val_f1, time.time() - t0,
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), str(best_ckpt))

            if self.early_stopper.step(val_f1):
                logger.info("Early stopping at epoch %d.", epoch)
                break

        self.model.load_state_dict(
            torch.load(str(best_ckpt), map_location=self.device, weights_only=True)
        )
        logger.info("Restored best checkpoint  val_f1=%.4f", best_val_f1)
        return history

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = total_correct = total_n = 0

        for batch in loader:
            self.optimizer.zero_grad()

            if self.is_multimodal:
                batch = [b.to(self.device) for b in batch]
                if len(batch) == 4:
                    x_eeg, x_env, adj, labels = batch
                    v_env_loss = None
                elif len(batch) == 5:
                    x_eeg, x_env, adj, v_env_loss, labels = batch
                else:
                    raise ValueError(
                        f"Expected 4 or 5 tensors for multimodal batch, got {len(batch)}"
                    )
                logits, v_eeg, v_env = self.model(x_eeg, x_env, adj)
                loss_fn = getattr(self.model, self.loss_name)
                loss_env = v_env if v_env_loss is None else v_env_loss
                loss, _, _ = loss_fn(
                    logits, labels, v_eeg, loss_env, **self.loss_kwargs
                )
            else:
                x_eeg, labels = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.model(x_eeg)
                loss_ce = nn.functional.cross_entropy(logits, labels)
                loss    = loss_ce

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            preds         = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_n       += labels.size(0)
            total_loss    += loss.item() * labels.size(0)

        return total_loss / total_n, total_correct / total_n

    def evaluate(self, loader: DataLoader) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        self.model.eval()
        all_logits, all_targets = [], []

        with torch.no_grad():
            for batch in loader:
                if self.is_multimodal:
                    batch = [b.to(self.device) for b in batch]
                    if len(batch) == 4:
                        x_eeg, x_env, adj, labels = batch
                    elif len(batch) == 5:
                        x_eeg, x_env, adj, _, labels = batch
                    else:
                        raise ValueError(
                            f"Expected 4 or 5 tensors for multimodal batch, got {len(batch)}"
                        )
                    logits, _, _ = self.model(x_eeg, x_env, adj)
                else:
                    x_eeg, labels = batch[0].to(self.device), batch[1].to(self.device)
                    logits = self.model(x_eeg)
                all_logits.append(logits.cpu())
                all_targets.append(labels.cpu())

        all_logits  = torch.cat(all_logits,  dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()

        preds = all_logits.argmax(axis=1)
        probs = torch.softmax(torch.tensor(all_logits), dim=-1).numpy()

        acc = float(accuracy_score(all_targets, preds))
        f1  = float(f1_score(all_targets, preds, average="macro", zero_division=0))
        try:
            auc = float(roc_auc_score(all_targets, probs,
                                      multi_class="ovr", average="macro"))
        except ValueError:
            auc = 0.5

        return acc, f1, auc, all_logits, all_targets
