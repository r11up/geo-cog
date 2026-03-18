"""Baseline models for the EEG-only, environment-only, and CNN ablation runs."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

from models.eeg_conformer import EEGConformer


class RandomForestBaseline:
    def __init__(self, n_estimators: int = 200, max_depth: int = 20,
                 random_state: int = 42):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion="gini",
            n_jobs=-1,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_flat = self._flatten_de(X)
        self.clf.fit(X_flat, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(self._flatten_de(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self._flatten_de(X))

    @staticmethod
    def _flatten_de(X: np.ndarray) -> np.ndarray:
        X_de = X[:, :, :5]
        return X_de.reshape(len(X_de), -1)


class BidirectionalLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 256,
        n_layers: int = 2,
        n_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        hidden_dim = out.shape[-1] // 2
        h_fwd = out[:, -1, :hidden_dim]
        h_bwd = out[:,  0, hidden_dim:]
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)
        return self.head(h_cat)


class EEGConformerOnly(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.encoder = EEGConformer(
            n_channels=cfg.preprocess.n_channels,
            n_features=10,
            temporal_kernel=m.eeg_temporal_kernel,
            n_temp_filters=40,
            depth_mult=2,
            pool_size=3,
            conformer_heads=m.conformer_heads,
            conformer_ff_dim=m.conformer_ff_dim,
            conformer_layers=m.conformer_layers,
            embed_dim=m.embed_dim,
            dropout=m.conformer_dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(cfg.model.clf_dropout),
            nn.Linear(m.embed_dim, cfg.eav.n_classes),
        )

    def forward(self, x_eeg: torch.Tensor) -> torch.Tensor:
        v_eeg = self.encoder(x_eeg)
        return self.head(v_eeg)


class EnvironmentOnlyMLP(nn.Module):
    def __init__(self, node_feat_dim: int = 6, n_classes: int = 5,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x_env: torch.Tensor) -> torch.Tensor:
        h = x_env.mean(dim=1)
        return self.net(h)


class EnvMLPEncoder(nn.Module):
    """Flat MLP over node features without graph adjacency."""

    def __init__(
        self,
        node_feat_dim: int = 6,
        n_nodes: int = 100,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_feat_dim * n_nodes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embed_dim),
        )

    def forward(self, F_graph: torch.Tensor, A_hat=None) -> torch.Tensor:
        B = F_graph.shape[0]
        x = F_graph.squeeze(1) if F_graph.dim() == 4 else F_graph
        return self.net(x.reshape(B, -1))


class EEGNetBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_channels = cfg.preprocess.n_channels
        n_classes = cfg.eav.n_classes
        f1 = 8
        depth_mult = 2
        f2 = f1 * depth_mult

        self.features = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(
                f1, f2, kernel_size=(n_channels, 1),
                groups=f1, bias=False,
            ),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(f2, f2, kernel_size=(1, 3), padding=(0, 1),
                      groups=f2, bias=False),
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(f2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        h = self.features(x).flatten(1)
        return self.classifier(h)
