"""
tests/test_models.py
====================
Unit tests for model components (models/).

Covers:
  - EEGConformer: forward pass shape, gradient flow
  - STGNN: forward pass shape with single and multi-timestep input
  - BiCrossModalAttention: output shape, bidirectionality
  - UniCrossModalAttention: output shape
  - NeuroGeospatialModel: end-to-end forward, loss computation
  - Baselines: BidirectionalLSTM, EEGConformerOnly, EnvironmentOnlyMLP
  - Loss: default squared-L2 alignment and optional InfoNCE alignment
  - RMA: fit → transform produces correct shapes

Run:
    pytest tests/test_models.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")

from config import Config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B      = 4      # batch size
N_CH   = 30
N_FEAT = 10     # DE + PSD
N_NODES = 100
ENV_DIM = 6


@pytest.fixture
def cfg():
    cfg = Config()
    # Smaller model for fast tests
    cfg.model.embed_dim        = 64
    cfg.model.conformer_heads  = 4
    cfg.model.conformer_ff_dim = 128
    cfg.model.conformer_layers = 1
    cfg.model.conformer_dropout = 0.0
    cfg.model.gcn_hidden       = 32
    cfg.model.gru_hidden       = 32
    cfg.model.fused_dim        = 128
    cfg.model.clf_hidden_dims  = [64]
    cfg.model.attn_heads       = 4
    cfg.model.lambda_align     = 0.1
    cfg.train.device           = "cpu"
    return cfg


@pytest.fixture
def eeg_batch():
    return torch.randn(B, N_CH, N_FEAT)


@pytest.fixture
def env_batch():
    return torch.randn(B, N_NODES, ENV_DIM)


@pytest.fixture
def adj():
    from data.geospatial import _make_grid_centroids, _queen_adjacency
    coords = _make_grid_centroids(10)
    A = _queen_adjacency(coords)
    return torch.from_numpy(A).float()


# ---------------------------------------------------------------------------
# EEGConformer
# ---------------------------------------------------------------------------

class TestEEGConformer:

    def test_forward_shape(self, cfg, eeg_batch):
        from models.eeg_conformer import EEGConformer
        model = EEGConformer(
            n_channels=N_CH, n_features=N_FEAT,
            embed_dim=cfg.model.embed_dim,
            conformer_heads=cfg.model.conformer_heads,
            conformer_ff_dim=cfg.model.conformer_ff_dim,
            conformer_layers=cfg.model.conformer_layers,
            dropout=0.0,
        )
        model.eval()
        with torch.no_grad():
            out = model(eeg_batch)
        assert out.shape == (B, cfg.model.embed_dim)

    def test_gradient_flows(self, cfg, eeg_batch):
        from models.eeg_conformer import EEGConformer
        model = EEGConformer(
            n_channels=N_CH, n_features=N_FEAT,
            embed_dim=cfg.model.embed_dim,
            conformer_heads=cfg.model.conformer_heads,
            conformer_ff_dim=cfg.model.conformer_ff_dim,
            conformer_layers=cfg.model.conformer_layers,
            dropout=0.0,
        )
        out = model(eeg_batch)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"

    def test_batch_size_1(self, cfg):
        from models.eeg_conformer import EEGConformer
        model = EEGConformer(
            n_channels=N_CH, n_features=N_FEAT,
            embed_dim=cfg.model.embed_dim,
            conformer_heads=cfg.model.conformer_heads,
            conformer_ff_dim=cfg.model.conformer_ff_dim,
            conformer_layers=1, dropout=0.0,
        )
        model.eval()
        x = torch.randn(1, N_CH, N_FEAT)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, cfg.model.embed_dim)

    def test_output_finite(self, cfg, eeg_batch):
        from models.eeg_conformer import EEGConformer
        model = EEGConformer(
            n_channels=N_CH, n_features=N_FEAT,
            embed_dim=cfg.model.embed_dim,
            conformer_heads=cfg.model.conformer_heads,
            conformer_ff_dim=cfg.model.conformer_ff_dim,
            conformer_layers=1, dropout=0.0,
        )
        model.eval()
        with torch.no_grad():
            out = model(eeg_batch)
        assert torch.all(torch.isfinite(out))


# ---------------------------------------------------------------------------
# STGNN
# ---------------------------------------------------------------------------

class TestSTGNN:

    def test_forward_single_timestep(self, cfg, env_batch, adj):
        from models.st_gnn import STGNN
        model = STGNN(
            node_feat_dim=ENV_DIM,
            gcn_hidden=cfg.model.gcn_hidden,
            gru_hidden=cfg.model.gru_hidden,
            embed_dim=cfg.model.embed_dim,
        )
        model.eval()
        with torch.no_grad():
            out = model(env_batch, adj)
        assert out.shape == (B, cfg.model.embed_dim)

    def test_forward_multi_timestep(self, cfg, adj):
        from models.st_gnn import STGNN
        model = STGNN(
            node_feat_dim=ENV_DIM,
            gcn_hidden=cfg.model.gcn_hidden,
            gru_hidden=cfg.model.gru_hidden,
            embed_dim=cfg.model.embed_dim,
            use_temporal=True,
        )
        T = 3
        x = torch.randn(B, T, N_NODES, ENV_DIM)
        model.eval()
        with torch.no_grad():
            out = model(x, adj)
        assert out.shape == (B, cfg.model.embed_dim)

    def test_default_forward_multi_timestep_uses_static_projection(self, cfg, adj):
        from models.st_gnn import STGNN
        model = STGNN(
            node_feat_dim=ENV_DIM,
            gcn_hidden=cfg.model.gcn_hidden,
            gru_hidden=cfg.model.gru_hidden,
            embed_dim=cfg.model.embed_dim,
        )
        T = 3
        x = torch.randn(B, T, N_NODES, ENV_DIM)
        model.eval()
        with torch.no_grad():
            out = model(x, adj)
        assert out.shape == (B, cfg.model.embed_dim)
        assert not hasattr(model, "gru")
        assert hasattr(model, "temporal_proj")

    def test_gradient_flows(self, cfg, env_batch, adj):
        from models.st_gnn import STGNN
        model = STGNN(
            node_feat_dim=ENV_DIM,
            gcn_hidden=cfg.model.gcn_hidden,
            gru_hidden=cfg.model.gru_hidden,
            embed_dim=cfg.model.embed_dim,
        )
        out = model(env_batch, adj)
        out.sum().backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"

    def test_output_finite(self, cfg, env_batch, adj):
        from models.st_gnn import STGNN
        model = STGNN(
            node_feat_dim=ENV_DIM,
            gcn_hidden=cfg.model.gcn_hidden,
            gru_hidden=cfg.model.gru_hidden,
            embed_dim=cfg.model.embed_dim,
        )
        model.eval()
        with torch.no_grad():
            out = model(env_batch, adj)
        assert torch.all(torch.isfinite(out))


# ---------------------------------------------------------------------------
# Bidirectional Cross-Modal Attention
# ---------------------------------------------------------------------------

class TestBiCrossModalAttention:

    def test_output_shapes(self, cfg):
        from models.fusion import BiCrossModalAttention
        attn = BiCrossModalAttention(
            embed_dim=cfg.model.embed_dim,
            n_heads=cfg.model.attn_heads,
            fused_dim=cfg.model.fused_dim,
        )
        v_eeg = torch.randn(B, cfg.model.embed_dim)
        v_env = torch.randn(B, cfg.model.embed_dim)
        z, A_eeg, A_env = attn(v_eeg, v_env)
        assert z.shape     == (B, cfg.model.fused_dim)
        assert A_eeg.shape == (B, cfg.model.embed_dim)
        assert A_env.shape == (B, cfg.model.embed_dim)

    def test_bidirectionality(self, cfg):
        """Swapping v_eeg ↔ v_env should change both attended representations."""
        from models.fusion import BiCrossModalAttention
        attn = BiCrossModalAttention(
            embed_dim=cfg.model.embed_dim,
            n_heads=cfg.model.attn_heads,
            fused_dim=cfg.model.fused_dim,
        )
        attn.eval()
        v_eeg = torch.randn(B, cfg.model.embed_dim)
        v_env = torch.randn(B, cfg.model.embed_dim)
        with torch.no_grad():
            z1, A_eeg1, A_env1 = attn(v_eeg, v_env)
            z2, A_eeg2, A_env2 = attn(v_env, v_eeg)   # swapped
        # Results must differ (independent attention directions)
        assert not torch.allclose(z1, z2, atol=1e-4)

    def test_all_projection_matrices_independent(self, cfg):
        """All 8 projection matrices must have distinct parameters."""
        from models.fusion import BiCrossModalAttention
        attn = BiCrossModalAttention(
            embed_dim=cfg.model.embed_dim,
            n_heads=cfg.model.attn_heads,
            fused_dim=cfg.model.fused_dim,
        )
        weights = [
            attn.W_Q_eeg.weight, attn.W_K_env.weight, attn.W_V_env.weight,
            attn.W_O_eeg.weight, attn.W_Q_env.weight, attn.W_K_eeg.weight,
            attn.W_V_eeg.weight, attn.W_O_env.weight,
        ]
        # All weight tensors must be distinct objects (separate parameters)
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                assert weights[i].data_ptr() != weights[j].data_ptr(), (
                    f"Weight {i} and {j} share the same parameter"
                )


class TestConcatFusion:

    def test_output_shapes(self, cfg):
        from models.fusion import ConcatFusion
        fusion = ConcatFusion(
            embed_dim=cfg.model.embed_dim,
            fused_dim=cfg.model.fused_dim,
        )
        v_eeg = torch.randn(B, cfg.model.embed_dim)
        v_env = torch.randn(B, cfg.model.embed_dim)
        z, eeg_out, env_out = fusion(v_eeg, v_env)
        assert z.shape == (B, cfg.model.fused_dim)
        assert eeg_out.shape == (B, cfg.model.embed_dim)
        assert env_out.shape == (B, cfg.model.embed_dim)


# ---------------------------------------------------------------------------
# Full NeuroGeospatialModel
# ---------------------------------------------------------------------------

class TestNeuroGeospatialModel:

    def test_proposed_model_uses_temporal_gru(self, cfg):
        from models.fusion import NeuroGeospatialModel
        model = NeuroGeospatialModel(cfg, bidirectional=True)
        assert model.env_encoder.use_temporal
        assert hasattr(model.env_encoder, "gru")
        assert not hasattr(model.env_encoder, "temporal_proj")

    def test_forward_bidirectional(self, cfg, eeg_batch, env_batch, adj):
        from models.fusion import NeuroGeospatialModel
        model = NeuroGeospatialModel(cfg, bidirectional=True)
        model.eval()
        with torch.no_grad():
            logits, v_eeg, v_env = model(eeg_batch, env_batch, adj)
        assert logits.shape == (B, cfg.eav.n_classes)
        assert v_eeg.shape  == (B, cfg.model.embed_dim)
        assert v_env.shape  == (B, cfg.model.embed_dim)

    def test_forward_concat_fusion(self, cfg, eeg_batch, env_batch, adj):
        from models.fusion import ConcatFusion, NeuroGeospatialModel
        model = NeuroGeospatialModel(cfg, bidirectional=True, fusion_type="concat")
        assert isinstance(model.fusion, ConcatFusion)
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(eeg_batch, env_batch, adj)
        assert logits.shape == (B, cfg.eav.n_classes)

    def test_forward_unidirectional(self, cfg, eeg_batch, env_batch, adj):
        from models.fusion import NeuroGeospatialModel
        model = NeuroGeospatialModel(cfg, bidirectional=False)
        model.eval()
        with torch.no_grad():
            logits, v_eeg, v_env = model(eeg_batch, env_batch, adj)
        assert logits.shape == (B, cfg.eav.n_classes)

    def test_alignment_loss_is_squared_l2_not_contrastive(self, cfg, eeg_batch, env_batch, adj):
        """
        Alignment loss must be squared L2 proximity between v_eeg and v_env.
        It is NOT a contrastive loss — verify by checking that identical
        embeddings produce zero alignment loss.
        """
        from models.fusion import NeuroGeospatialModel
        import torch.nn.functional as F
        model = NeuroGeospatialModel(cfg, bidirectional=True)
        labels = torch.zeros(B, dtype=torch.long)
        logits, v_eeg, v_env = model(eeg_batch, env_batch, adj)

        # When embeddings are forced equal, alignment loss must be ~0
        loss_total, loss_ce, loss_align = model.compute_loss(
            logits, labels, v_eeg, v_eeg   # identical embeddings
        )
        assert loss_align.item() < 1e-6, (
            f"Alignment loss with identical embeddings should be ~0, got {loss_align.item()}"
        )

        offset = torch.ones_like(v_eeg)
        _, _, loss_align_offset = model.compute_loss(logits, labels, v_eeg, v_eeg + offset)
        assert torch.isclose(
            loss_align_offset,
            torch.tensor(float(cfg.model.embed_dim), device=loss_align_offset.device),
            atol=1e-4,
        )

    def test_total_loss_components(self, cfg, eeg_batch, env_batch, adj):
        from models.fusion import NeuroGeospatialModel
        model = NeuroGeospatialModel(cfg, bidirectional=True)
        labels = torch.randint(0, cfg.eav.n_classes, (B,))
        logits, v_eeg, v_env = model(eeg_batch, env_batch, adj)
        loss_total, loss_ce, loss_align = model.compute_loss(logits, labels, v_eeg, v_env)

        # L_total = L_CE + λ * L_align
        expected = loss_ce + cfg.model.lambda_align * loss_align
        assert torch.isclose(loss_total, expected, atol=1e-5)

    def test_infonce_loss_components(self, cfg, eeg_batch, env_batch, adj):
        from models.fusion import NeuroGeospatialModel
        model = NeuroGeospatialModel(cfg, bidirectional=True)
        labels = torch.randint(0, cfg.eav.n_classes, (B,))
        logits, v_eeg, v_env = model(eeg_batch, env_batch, adj)
        loss_total, loss_ce, loss_align = model.compute_loss_infonce(
            logits, labels, v_eeg, v_env, temperature=0.2
        )
        expected = loss_ce + cfg.model.lambda_align * loss_align
        assert torch.isclose(loss_total, expected, atol=1e-5)
        assert torch.isfinite(loss_align)

    def test_gradient_flows_end_to_end(self, cfg, eeg_batch, env_batch, adj):
        from models.fusion import NeuroGeospatialModel
        model = NeuroGeospatialModel(cfg, bidirectional=True)
        labels = torch.randint(0, cfg.eav.n_classes, (B,))
        logits, v_eeg, v_env = model(eeg_batch, env_batch, adj)
        loss, _, _ = model.compute_loss(logits, labels, v_eeg, v_env)
        loss.backward()
        n_params_with_grad = sum(
            1 for p in model.parameters() if p.requires_grad and p.grad is not None
        )
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        assert n_params_with_grad == total_params, (
            f"Only {n_params_with_grad}/{total_params} params received gradients"
        )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class TestBaselines:

    def test_bilstm_shape(self, cfg):
        from models.baselines import BidirectionalLSTM
        model = BidirectionalLSTM(
            input_dim=N_FEAT, hidden_dim=32, n_layers=2,
            n_classes=cfg.eav.n_classes,
        )
        model.eval()
        x = torch.randn(B, N_CH, N_FEAT)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, cfg.eav.n_classes)

    def test_bilstm_uses_both_directions(self, cfg):
        """BidirectionalLSTM forward() must concatenate h_fwd and h_bwd."""
        import inspect
        from models.baselines import BidirectionalLSTM
        src = inspect.getsource(BidirectionalLSTM.forward)
        assert 'h_fwd' in src and 'h_bwd' in src, (
            "BiLSTM must use both forward (h_fwd) and backward (h_bwd) states"
        )

    def test_eeg_conformer_only_shape(self, cfg, eeg_batch):
        from models.baselines import EEGConformerOnly
        model = EEGConformerOnly(cfg)
        model.eval()
        with torch.no_grad():
            out = model(eeg_batch)
        assert out.shape == (B, cfg.eav.n_classes)

    def test_env_mlp_shape(self, cfg, env_batch):
        from models.baselines import EnvironmentOnlyMLP
        model = EnvironmentOnlyMLP(node_feat_dim=ENV_DIM, n_classes=cfg.eav.n_classes)
        model.eval()
        with torch.no_grad():
            out = model(env_batch)
        assert out.shape == (B, cfg.eav.n_classes)

    def test_env_mlp_encoder_shape(self, cfg, env_batch, adj):
        from models.baselines import EnvMLPEncoder
        model = EnvMLPEncoder(
            node_feat_dim=ENV_DIM,
            n_nodes=N_NODES,
            embed_dim=cfg.model.embed_dim,
        )
        model.eval()
        with torch.no_grad():
            out = model(env_batch, adj)
        assert out.shape == (B, cfg.model.embed_dim)

    def test_eegnet_shape(self, cfg, eeg_batch):
        from models.baselines import EEGNetBaseline
        model = EEGNetBaseline(cfg)
        model.eval()
        with torch.no_grad():
            out = model(eeg_batch)
        assert out.shape == (B, cfg.eav.n_classes)


# ---------------------------------------------------------------------------
# RMA alignment (alignment.py)
# ---------------------------------------------------------------------------

class TestRMA:

    def test_fit_transform_shapes(self):
        from data.alignment import RMA, compute_eeg_covariances, PECM
        from data.geospatial import _grids_to_node_features
        import numpy as np

        rng   = np.random.default_rng(7)
        g = rng.uniform(0, 1, (10, 10)).astype(np.float32)
        feats = _grids_to_node_features({
            "pm25_grid": g * 50 + 15,
            "no2_grid":  g * 30 + 10,
            "ndvi_grid": (1 - g) * 0.5,
            "far_grid":  g * 5,
            "gvi_grid":  (1 - g) * 0.7,
            "o3_grid":   (1 - g) * 40 + 40,
        }, 10)
        pecm  = PECM(feats, random_state=42)
        labels = np.tile(np.arange(5), 10)
        env   = pecm.sample(labels)

        raw = rng.standard_normal((50, 30, 500)).astype(np.float32)
        covs = compute_eeg_covariances(raw)
        assert covs.shape == (50, 30, 30)

        # All covariances must be SPD (positive eigenvalues)
        for i in range(50):
            eigs = np.linalg.eigvalsh(covs[i])
            assert (eigs > 0).all(), f"Cov {i} is not positive definite"

        rma = RMA(embed_dim=256)
        Z_eeg, Z_env = rma.fit(covs, env)
        assert Z_eeg.shape == (50, 256)
        assert Z_env.shape == (50, 256)

        Z_eeg2, Z_env2 = rma.transform(covs, env)
        assert Z_eeg2.shape == (50, 256)
        assert Z_env2.shape == (50, 256)

    def test_procrustes_not_transposed(self):
        """R_hat (not R_hat.T) must be applied in both fit and transform."""
        import inspect
        from data import alignment as al_mod
        src = inspect.getsource(al_mod.RMA)
        # Get only non-comment lines
        code_lines = [l for l in src.splitlines() if not l.strip().startswith('#')]
        r_hat_T_in_code = any('R_hat.T' in l for l in code_lines)
        assert not r_hat_T_in_code, "R_hat.T found in RMA code — should be R_hat"
