"""Neural and environmental encoders are fused, then trained with supervised and embedding-alignment losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from models.eeg_conformer import EEGConformer
from models.st_gnn import STGNN


class BiCrossModalAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        n_heads: int = 8,
        fused_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dk = embed_dim // n_heads

        self.W_Q_eeg  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K_env  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V_env  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O_eeg  = nn.Linear(embed_dim, embed_dim, bias=False)

        self.W_Q_env  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K_eeg  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V_eeg  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O_env  = nn.Linear(embed_dim, embed_dim, bias=False)

        self.fusion_proj = nn.Linear(2 * embed_dim, fused_dim, bias=True)

        self.dropout = nn.Dropout(p=dropout)
        self.fused_dim = fused_dim

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return x.view(B, self.n_heads, self.dk)

    def _mha(self, Q: torch.Tensor, K: torch.Tensor,
             V: torch.Tensor, W_O: nn.Linear) -> torch.Tensor:
        B = Q.shape[0]
        Q_h = self._split_heads(Q)
        K_h = self._split_heads(K)
        V_h = self._split_heads(V)

        scale = self.dk ** -0.5
        attn_logits = torch.matmul(Q_h, K_h.transpose(1, 2)) * scale
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attended = torch.matmul(attn_weights, V_h)
        attended = attended.reshape(B, self.embed_dim)

        out = W_O(attended)
        return out

    def forward(
        self,
        v_eeg: torch.Tensor,
        v_env: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Q_eeg = self.W_Q_eeg(v_eeg)
        K_env = self.W_K_env(v_env)
        V_env = self.W_V_env(v_env)
        A_eeg = self._mha(Q_eeg, K_env, V_env, self.W_O_eeg)

        Q_env = self.W_Q_env(v_env)
        K_eeg = self.W_K_eeg(v_eeg)
        V_eeg = self.W_V_eeg(v_eeg)
        A_env = self._mha(Q_env, K_eeg, V_eeg, self.W_O_env)

        concat = torch.cat([A_eeg, A_env], dim=-1)
        z_fused = self.fusion_proj(concat)

        return z_fused, A_eeg, A_env


class ConcatFusion(nn.Module):
    def __init__(self, embed_dim: int = 256, fused_dim: int = 512):
        super().__init__()
        self.fusion_proj = nn.Linear(2 * embed_dim, fused_dim, bias=True)
        self.fused_dim = fused_dim

    def forward(
        self,
        v_eeg: torch.Tensor,
        v_env: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_fused = self.fusion_proj(torch.cat([v_eeg, v_env], dim=-1))
        return z_fused, v_eeg, v_env


class UniCrossModalAttention(nn.Module):
    def __init__(self, embed_dim: int = 256, n_heads: int = 8,
                 fused_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dk = embed_dim // n_heads

        self.W_Q_eeg = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K_env = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V_env = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O_eeg = nn.Linear(embed_dim, embed_dim, bias=False)

        self.fusion_proj = nn.Linear(embed_dim, fused_dim, bias=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fused_dim = fused_dim

    def _split_heads(self, x):
        B = x.shape[0]
        return x.view(B, self.n_heads, self.dk)

    def forward(self, v_eeg, v_env):
        B = v_eeg.shape[0]
        Q_eeg = self.W_Q_eeg(v_eeg)
        K_env = self.W_K_env(v_env)
        V_env = self.W_V_env(v_env)

        Q_h = self._split_heads(Q_eeg)
        K_h = self._split_heads(K_env)
        V_h = self._split_heads(V_env)

        scale = self.dk ** -0.5
        attn_w = torch.softmax(
            torch.matmul(Q_h, K_h.transpose(1, 2)) * scale, dim=-1
        )
        attn_w = self.dropout(attn_w)
        attended = torch.matmul(attn_w, V_h).reshape(B, self.embed_dim)
        A_eeg = self.W_O_eeg(attended)
        z_fused = self.fusion_proj(A_eeg)
        return z_fused, A_eeg, None


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int = 512,
                 hidden_dims=(256,),
                 n_classes: int = 5,
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class NeuroGeospatialModel(nn.Module):
    def __init__(
        self,
        cfg,
        bidirectional: bool = True,
        fusion_type: Optional[str] = None,
    ):
        super().__init__()

        m = cfg.model
        e = cfg.eav

        self.eeg_encoder = EEGConformer(
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

        self.env_encoder = STGNN(
            node_feat_dim=cfg.geo.node_feature_dim,
            gcn_intermediate=m.gcn_intermediate,
            gcn_hidden=m.gcn_hidden,
            gru_hidden=m.gru_hidden,
            gcn_layers=m.gcn_layers,
            gru_layers=m.gru_layers,
            embed_dim=m.embed_dim,
            use_temporal=True,
        )

        fusion_type = fusion_type or ("bidirectional" if bidirectional else "unidirectional")
        fusion_type = fusion_type.lower()

        if fusion_type in {"bidirectional", "bidir"}:
            self.fusion = BiCrossModalAttention(
                embed_dim=m.embed_dim,
                n_heads=m.attn_heads,
                fused_dim=m.fused_dim,
            )
        elif fusion_type in {"unidirectional", "uni"}:
            self.fusion = UniCrossModalAttention(
                embed_dim=m.embed_dim,
                n_heads=m.attn_heads,
                fused_dim=m.fused_dim,
            )
        elif fusion_type in {"concat", "concatenation"}:
            self.fusion = ConcatFusion(
                embed_dim=m.embed_dim,
                fused_dim=m.fused_dim,
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type!r}")

        self.classifier = MLPClassifier(
            in_dim=m.fused_dim,
            hidden_dims=tuple(m.clf_hidden_dims),
            n_classes=e.n_classes,
            dropout=m.clf_dropout,
        )

        self.lambda_align = m.lambda_align

    def forward(
        self,
        x_eeg: torch.Tensor,
        x_env: torch.Tensor,
        adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v_eeg = self.eeg_encoder(x_eeg)
        v_env = self.env_encoder(x_env, adj)
        z_fused, _, _ = self.fusion(v_eeg, v_env)
        logits = self.classifier(z_fused)
        return logits, v_eeg, v_env

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        v_eeg: torch.Tensor,
        v_env: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_ce = F.cross_entropy(logits, targets)
        diff = v_eeg - v_env
        loss_align = (diff * diff).sum(dim=-1).mean()
        loss_total = loss_ce + self.lambda_align * loss_align
        return loss_total, loss_ce, loss_align

    def compute_loss_infonce(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        v_eeg: torch.Tensor,
        v_env: torch.Tensor,
        temperature: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        loss_ce = F.cross_entropy(logits, targets)
        eeg_n = F.normalize(v_eeg, dim=-1)
        env_n = F.normalize(v_env, dim=-1)
        sim = eeg_n @ env_n.T / temperature
        labels = torch.arange(sim.shape[0], device=sim.device)
        loss_align = F.cross_entropy(sim, labels)
        loss_total = loss_ce + self.lambda_align * loss_align
        return loss_total, loss_ce, loss_align
