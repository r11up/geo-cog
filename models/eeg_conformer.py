"""EEG-Conformer encoder for the neural feature tensor."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):
    def __init__(self, n_filters: int = 40, kernel_size: int = 25,
                 n_features: int = 10):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(n_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.elu(x)
        return x


class SpatialDepthwiseConvBlock(nn.Module):
    def __init__(self, n_filters: int = 40, n_channels: int = 30,
                 depth_mult: int = 2, pool_size: int = 3, dropout: float = 0.5):
        super().__init__()
        out_ch = n_filters * depth_mult
        self.depthwise = nn.Conv2d(
            in_channels=n_filters,
            out_channels=out_ch,
            kernel_size=(n_channels, 1),
            groups=n_filters,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_size))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn(x)
        x = F.elu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class ConformerTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8,
                 ff_dim: int = 512, n_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class EEGConformer(nn.Module):
    def __init__(
        self,
        n_channels: int = 30,
        n_features: int = 10,
        temporal_kernel: int = 25,
        n_temp_filters: int = 40,
        depth_mult: int = 2,
        pool_size: int = 3,
        conformer_heads: int = 8,
        conformer_ff_dim: int = 512,
        conformer_layers: int = 2,
        embed_dim: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.temporal_conv = TemporalConvBlock(
            n_filters=n_temp_filters,
            kernel_size=temporal_kernel,
            n_features=n_features,
        )
        self.spatial_conv = SpatialDepthwiseConvBlock(
            n_filters=n_temp_filters,
            n_channels=n_channels,
            depth_mult=depth_mult,
            pool_size=pool_size,
            dropout=dropout,
        )

        patched_len = max(1, n_features // pool_size)
        d_model = n_temp_filters * depth_mult

        self.transformer = ConformerTransformerEncoder(
            d_model=d_model,
            n_heads=conformer_heads,
            ff_dim=conformer_ff_dim,
            n_layers=conformer_layers,
            dropout=dropout,
        )

        self.proj = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(patched_len * d_model, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self._patched_len = patched_len
        self._d_model = d_model
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F = x.shape
        x = x.unsqueeze(1)

        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)

        x = self.transformer(x)

        v_eeg = self.proj(x)
        return v_eeg
