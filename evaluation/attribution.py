"""Estimate environmental feature attributions with a lightweight GraphLIME-style perturbation method."""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

ENV_FEAT_NAMES = ["PM2.5", "NDVI", "FAR", "GVI", "NO2", "O3"]


class GraphLIMEAttributor:
    def __init__(self, model, device: str, n_classes: int = 5):
        self.model = model
        self.device = device
        self.n_classes = n_classes
        self.model.eval()

    def attribute(
        self,
        X_eeg: np.ndarray,
        node_feats: np.ndarray,
        adj: np.ndarray,
        class_ids: np.ndarray,
        n_perturb: int = 200,
        sigma: float = 0.1,
        seed: int = 42,
    ) -> np.ndarray:
        """Return mean absolute feature attribution grouped by the supplied class IDs."""
        rng = np.random.default_rng(seed)
        n_feats = node_feats.shape[1]

        class_attributions: Dict[int, List[np.ndarray]] = {c: [] for c in range(self.n_classes)}

        adj_t     = torch.from_numpy(adj).float().to(self.device)

        for i in range(len(X_eeg)):
            x_eeg_i = torch.from_numpy(X_eeg[i:i+1]).float().to(self.device)
            cls = int(class_ids[i])

            base_nf = torch.from_numpy(node_feats).float().to(self.device).unsqueeze(0)
            with torch.no_grad():
                logits_base, _, _ = self.model(x_eeg_i, base_nf, adj_t)
                prob_base = F.softmax(logits_base, dim=-1)[0, cls].item()

            delta_probs = np.zeros((n_perturb, n_feats))
            perturbs    = np.zeros((n_perturb, n_feats))

            for s in range(n_perturb):
                perturb = rng.normal(0, sigma, size=(1, n_feats))
                perturbs[s] = perturb[0]
                nf_perturbed = node_feats + perturb
                nf_t = torch.from_numpy(nf_perturbed.astype(np.float32)).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    logits_p, _, _ = self.model(x_eeg_i, nf_t, adj_t)
                    prob_p = F.softmax(logits_p, dim=-1)[0, cls].item()
                delta_probs[s] = prob_p - prob_base

            weights = np.exp(-np.sum(perturbs ** 2, axis=1) / (sigma ** 2))
            W_diag = np.diag(weights)

            P = perturbs
            lhs = P.T @ W_diag @ P + 1e-6 * np.eye(n_feats)
            rhs = P.T @ W_diag @ delta_probs.mean(axis=1)
            try:
                beta = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                beta = np.zeros(n_feats)

            class_attributions[cls].append(np.abs(beta))

        attribution = np.zeros((self.n_classes, n_feats))
        for c in range(self.n_classes):
            if class_attributions[c]:
                attribution[c] = np.mean(class_attributions[c], axis=0)

        return attribution

    def rank_features(
        self, attribution: np.ndarray, class_names: Tuple[str, ...] = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        if class_names is None:
            class_names = tuple(f"Class_{i}" for i in range(attribution.shape[0]))
        rankings: Dict = {}
        for c, cls_name in enumerate(class_names):
            scores = attribution[c]
            ranked = sorted(
                zip(ENV_FEAT_NAMES, scores.tolist()),
                key=lambda x: x[1], reverse=True,
            )
            rankings[cls_name] = ranked
        return rankings
