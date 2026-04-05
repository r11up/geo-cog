"""Subject-level cross-validation for Astana runs and the Singapore environmental-domain swap."""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold

from data.alignment import PECM
from data.geospatial import build_astana_grid, build_singapore_grid
from training.trainer import Trainer

logger = logging.getLogger(__name__)


def _sample_env_without_labels(pecm: PECM, n_samples: int) -> np.ndarray:
    return pecm.sample_unconditional(n_samples)


class EEGOnlyDataset(Dataset):
    def __init__(self, X: np.ndarray, labels: np.ndarray):
        self.X      = torch.from_numpy(X).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]


class MultimodalDataset(Dataset):
    def __init__(
        self,
        X:          np.ndarray,
        env_vecs:   np.ndarray,
        node_feats: np.ndarray,
        adj:        np.ndarray,
        labels:     np.ndarray,
    ):
        self.X          = torch.from_numpy(X).float()
        self.env_vecs   = torch.from_numpy(env_vecs).float()
        self.node_feats = torch.from_numpy(node_feats).float()
        self.adj        = torch.from_numpy(adj).float()
        self.labels     = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x_env = self.node_feats + self.env_vecs[idx].unsqueeze(0)
        return self.X[idx], x_env, self.adj, self.labels[idx]


class MultimodalAlignmentDataset(MultimodalDataset):
    def __init__(
        self,
        X:              np.ndarray,
        env_vecs:       np.ndarray,
        node_feats:     np.ndarray,
        adj:            np.ndarray,
        align_env_vecs: np.ndarray,
        labels:         np.ndarray,
    ):
        super().__init__(X, env_vecs, node_feats, adj, labels)
        self.align_env_vecs = torch.from_numpy(align_env_vecs).float()

    def __getitem__(self, idx):
        x_eeg, x_env, adj, label = super().__getitem__(idx)
        return x_eeg, x_env, adj, self.align_env_vecs[idx], label


class _EnvOnlyDataset(Dataset):
    def __init__(self, env_vecs, node_feats, adj, labels):
        self.env_vecs   = torch.from_numpy(env_vecs).float()
        self.node_feats = torch.from_numpy(node_feats).float()
        self.adj        = torch.from_numpy(adj).float()
        self.labels     = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x_env = self.node_feats + self.env_vecs[idx].unsqueeze(0)
        return x_env, self.labels[idx]


def run_subject_cv(
    X:             np.ndarray,
    labels:        np.ndarray,
    subject_ids:   np.ndarray,
    cfg,
    model_factory: Callable,
    mode:          str = "multimodal",
    epochs_raw:    Optional[np.ndarray] = None,
    city:          str = "astana",
    fold_callback: Optional[Callable[..., None]] = None,
) -> Tuple[Dict[str, List[float]], List[Dict]]:
    use_cuda = cfg.train.device.startswith("cuda")
    city = city.lower()

    if city == "astana":
        node_feats, adj = build_astana_grid(
            cfg,
            gee_project=getattr(cfg.geo, "gee_project", None),
            openaq_api_key=getattr(cfg.geo, "openaq_api_key", None),
            year=getattr(cfg.geo, "satellite_year", 2025),
        )
    elif city == "singapore":
        node_feats, adj = build_singapore_grid(
            cfg,
            gee_project=getattr(cfg.geo, "gee_project", None),
            openaq_api_key=getattr(cfg.geo, "openaq_api_key", None),
            year=getattr(cfg.geo, "satellite_year", 2025),
        )
    else:
        raise ValueError(f"Unknown city '{city}'.")

    logger.info("run_subject_cv — city=%s  mode=%s  n=%d",
                city.title(), mode, len(labels))

    fold_metrics = {"accuracy": [], "macro_f1": [], "roc_auc": []}
    histories    = []

    for fold_id, (train_val_idx, test_idx) in enumerate(
        GroupKFold(n_splits=cfg.train.n_folds).split(X, labels, groups=subject_ids)
    ):
        logger.info("FOLD %d / %d", fold_id + 1, cfg.train.n_folds)

        tv_subjects    = np.unique(subject_ids[train_val_idx])
        val_subjects   = tv_subjects[-5:]
        train_subjects = tv_subjects[:-5]
        train_mask = np.isin(subject_ids[train_val_idx], train_subjects)
        val_mask   = np.isin(subject_ids[train_val_idx], val_subjects)
        train_idx  = train_val_idx[train_mask]
        val_idx    = train_val_idx[val_mask]

        X_tr, y_tr = X[train_idx], labels[train_idx]
        X_va, y_va = X[val_idx],   labels[val_idx]
        X_te, y_te = X[test_idx],  labels[test_idx]

        logger.info("  train=%d  val=%d  test=%d",
                    len(train_idx), len(val_idx), len(test_idx))

        if mode == "eeg_only":
            train_ds      = EEGOnlyDataset(X_tr, y_tr)
            val_ds        = EEGOnlyDataset(X_va, y_va)
            test_ds       = EEGOnlyDataset(X_te, y_te)
            is_multimodal = False

        elif mode == "multimodal":
            pecm     = PECM(node_feats, random_state=cfg.train.random_seed)
            env_tr   = pecm.sample(y_tr)
            env_va   = _sample_env_without_labels(pecm, len(y_va))
            env_te   = _sample_env_without_labels(pecm, len(y_te))
            train_ds = MultimodalDataset(X_tr, env_tr, node_feats, adj, y_tr)
            val_ds   = MultimodalDataset(X_va, env_va, node_feats, adj, y_va)
            test_ds  = MultimodalDataset(X_te, env_te, node_feats, adj, y_te)
            is_multimodal = True

        elif mode == "multimodal_oracle_test":
            pecm     = PECM(node_feats, random_state=cfg.train.random_seed)
            env_tr   = pecm.sample(y_tr)
            env_va   = pecm.sample(y_va)
            env_te   = pecm.sample(y_te)
            train_ds = MultimodalDataset(X_tr, env_tr, node_feats, adj, y_tr)
            val_ds   = MultimodalDataset(X_va, env_va, node_feats, adj, y_va)
            test_ds  = MultimodalDataset(X_te, env_te, node_feats, adj, y_te)
            is_multimodal = True

        elif mode == "env_only":
            pecm     = PECM(node_feats, random_state=cfg.train.random_seed)
            env_tr   = pecm.sample(y_tr)
            env_va   = _sample_env_without_labels(pecm, len(y_va))
            env_te   = _sample_env_without_labels(pecm, len(y_te))
            train_ds = _EnvOnlyDataset(env_tr, node_feats, adj, y_tr)
            val_ds   = _EnvOnlyDataset(env_va, node_feats, adj, y_va)
            test_ds  = _EnvOnlyDataset(env_te, node_feats, adj, y_te)
            is_multimodal = False

        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        bs = cfg.train.batch_size
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=0, pin_memory=use_cuda)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                                  num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                                  num_workers=0)

        model   = model_factory()
        trainer = Trainer(model, cfg, cfg.train.device, is_multimodal=is_multimodal)
        history = trainer.train(train_loader, val_loader, fold_id=fold_id)
        histories.append(history)

        acc, f1, auc, logits, targets = trainer.evaluate(test_loader)
        logger.info("  Fold %d  acc=%.4f  f1=%.4f  auc=%.4f",
                    fold_id + 1, acc, f1, auc)

        fold_metrics["accuracy"].append(acc)
        fold_metrics["macro_f1"].append(f1)
        fold_metrics["roc_auc"].append(auc)

        if fold_callback is not None:
            fold_callback(
                fold_id=fold_id,
                model=trainer.model,
                node_feats=node_feats,
                adj=adj,
                X_test=X_te,
                y_test=y_te,
                logits=logits,
                targets=targets,
                mode=mode,
                city=city,
            )

    _log_cv_summary(fold_metrics)
    return fold_metrics, histories


def _build_astana_graph_for_ablation(cfg):
    return build_astana_grid(
        cfg,
        gee_project=getattr(cfg.geo, "gee_project", None),
        openaq_api_key=getattr(cfg.geo, "openaq_api_key", None),
        year=getattr(cfg.geo, "satellite_year", 2025),
    )


def _make_stgnn(cfg, use_temporal: bool):
    from models.st_gnn import STGNN

    return STGNN(
        node_feat_dim=cfg.geo.node_feature_dim,
        gcn_intermediate=cfg.model.gcn_intermediate,
        gcn_hidden=cfg.model.gcn_hidden,
        gcn_layers=cfg.model.gcn_layers,
        gru_hidden=cfg.model.gru_hidden,
        gru_layers=cfg.model.gru_layers,
        embed_dim=cfg.model.embed_dim,
        use_temporal=use_temporal,
    )


def _iter_subject_folds(X, labels, subject_ids, cfg):
    for fold_id, (train_val_idx, test_idx) in enumerate(
        GroupKFold(cfg.train.n_folds).split(X, labels, groups=subject_ids)
    ):
        tv_subjects = np.unique(subject_ids[train_val_idx])
        val_subjects = tv_subjects[-5:]
        train_idx = train_val_idx[np.isin(subject_ids[train_val_idx], tv_subjects[:-5])]
        val_idx = train_val_idx[np.isin(subject_ids[train_val_idx], val_subjects)]
        yield fold_id, train_idx, val_idx, test_idx


def _run_multimodal_ablation(
    X: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    cfg,
    build_model: Callable[[np.ndarray, np.ndarray], torch.nn.Module],
    train_env_sampler: Callable[[PECM, np.ndarray, np.ndarray, int], np.ndarray],
    model_initializer: Optional[Callable] = None,
    pecm_factory: Optional[Callable[[np.ndarray], PECM]] = None,
    trainer_kwargs: Optional[Dict] = None,
) -> Tuple[Dict[str, List[float]], List[Dict]]:
    node_feats, adj = _build_astana_graph_for_ablation(cfg)
    use_cuda = cfg.train.device.startswith("cuda")
    fold_metrics = {"accuracy": [], "macro_f1": [], "roc_auc": []}
    histories = []
    trainer_kwargs = trainer_kwargs or {}

    if pecm_factory is None:
        pecm_factory = lambda feats: PECM(feats, random_state=cfg.train.random_seed)

    for fold_id, train_idx, val_idx, test_idx in _iter_subject_folds(
        X, labels, subject_ids, cfg
    ):
        pecm = pecm_factory(node_feats)
        env_tr = train_env_sampler(pecm, labels[train_idx], train_idx, fold_id)
        env_va = pecm.sample_unconditional(len(val_idx))
        env_te = pecm.sample_unconditional(len(test_idx))

        bs = cfg.train.batch_size
        train_loader = DataLoader(
            MultimodalDataset(X[train_idx], env_tr, node_feats, adj, labels[train_idx]),
            batch_size=bs, shuffle=True, num_workers=0, pin_memory=use_cuda,
        )
        val_loader = DataLoader(
            MultimodalDataset(X[val_idx], env_va, node_feats, adj, labels[val_idx]),
            batch_size=bs, shuffle=False, num_workers=0,
        )
        test_loader = DataLoader(
            MultimodalDataset(X[test_idx], env_te, node_feats, adj, labels[test_idx]),
            batch_size=bs, shuffle=False, num_workers=0,
        )

        model = build_model(node_feats, adj)
        if model_initializer is not None:
            model = model_initializer(model)
        trainer = Trainer(
            model,
            cfg,
            cfg.train.device,
            is_multimodal=True,
            **trainer_kwargs,
        )
        histories.append(trainer.train(train_loader, val_loader, fold_id=fold_id))

        acc, f1, auc, _, _ = trainer.evaluate(test_loader)
        fold_metrics["accuracy"].append(acc)
        fold_metrics["macro_f1"].append(f1)
        fold_metrics["roc_auc"].append(auc)

    _log_cv_summary(fold_metrics)
    return fold_metrics, histories


def run_no_alignment_ablation(
    X: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    cfg,
    model_initializer: Optional[Callable] = None,
) -> Tuple[Dict[str, List[float]], List[Dict]]:
    from models.fusion import NeuroGeospatialModel

    node_feats, adj = _build_astana_graph_for_ablation(cfg)
    use_cuda = cfg.train.device.startswith("cuda")
    fold_metrics = {"accuracy": [], "macro_f1": [], "roc_auc": []}
    histories = []

    for fold_id, (train_val_idx, test_idx) in enumerate(
        GroupKFold(cfg.train.n_folds).split(X, labels, groups=subject_ids)
    ):
        tv_subjects = np.unique(subject_ids[train_val_idx])
        val_subjects = tv_subjects[-5:]
        train_idx = train_val_idx[np.isin(subject_ids[train_val_idx], tv_subjects[:-5])]
        val_idx = train_val_idx[np.isin(subject_ids[train_val_idx], val_subjects)]

        pecm = PECM(node_feats, random_state=cfg.train.random_seed)
        env_tr = pecm.sample_unconditional(len(train_idx))
        env_va = pecm.sample_unconditional(len(val_idx))
        env_te = pecm.sample_unconditional(len(test_idx))

        bs = cfg.train.batch_size
        train_loader = DataLoader(
            MultimodalDataset(X[train_idx], env_tr, node_feats, adj, labels[train_idx]),
            batch_size=bs, shuffle=True, num_workers=0, pin_memory=use_cuda,
        )
        val_loader = DataLoader(
            MultimodalDataset(X[val_idx], env_va, node_feats, adj, labels[val_idx]),
            batch_size=bs, shuffle=False, num_workers=0,
        )
        test_loader = DataLoader(
            MultimodalDataset(X[test_idx], env_te, node_feats, adj, labels[test_idx]),
            batch_size=bs, shuffle=False, num_workers=0,
        )

        model = NeuroGeospatialModel(cfg, bidirectional=True)
        if model_initializer is not None:
            model = model_initializer(model)
        trainer = Trainer(model, cfg, cfg.train.device, is_multimodal=True)
        histories.append(trainer.train(train_loader, val_loader, fold_id=fold_id))

        acc, f1, auc, _, _ = trainer.evaluate(test_loader)
        fold_metrics["accuracy"].append(acc)
        fold_metrics["macro_f1"].append(f1)
        fold_metrics["roc_auc"].append(auc)

    _log_cv_summary(fold_metrics)
    return fold_metrics, histories


def run_env_mlp_ablation(
    X: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    cfg,
    model_initializer: Optional[Callable] = None,
) -> Tuple[Dict[str, List[float]], List[Dict]]:
    from models.fusion import NeuroGeospatialModel
    from models.baselines import EnvMLPEncoder

    node_feats, adj = _build_astana_graph_for_ablation(cfg)
    use_cuda = cfg.train.device.startswith("cuda")
    fold_metrics = {"accuracy": [], "macro_f1": [], "roc_auc": []}
    histories = []

    for fold_id, (train_val_idx, test_idx) in enumerate(
        GroupKFold(cfg.train.n_folds).split(X, labels, groups=subject_ids)
    ):
        tv_subjects = np.unique(subject_ids[train_val_idx])
        val_subjects = tv_subjects[-5:]
        train_idx = train_val_idx[np.isin(subject_ids[train_val_idx], tv_subjects[:-5])]
        val_idx = train_val_idx[np.isin(subject_ids[train_val_idx], val_subjects)]

        pecm = PECM(node_feats, random_state=cfg.train.random_seed)
        env_tr = pecm.sample(labels[train_idx])
        env_va = pecm.sample_unconditional(len(val_idx))
        env_te = pecm.sample_unconditional(len(test_idx))

        bs = cfg.train.batch_size
        train_loader = DataLoader(
            MultimodalDataset(X[train_idx], env_tr, node_feats, adj, labels[train_idx]),
            batch_size=bs, shuffle=True, num_workers=0, pin_memory=use_cuda,
        )
        val_loader = DataLoader(
            MultimodalDataset(X[val_idx], env_va, node_feats, adj, labels[val_idx]),
            batch_size=bs, shuffle=False, num_workers=0,
        )
        test_loader = DataLoader(
            MultimodalDataset(X[test_idx], env_te, node_feats, adj, labels[test_idx]),
            batch_size=bs, shuffle=False, num_workers=0,
        )

        model = NeuroGeospatialModel(cfg, bidirectional=True)
        model.env_encoder = EnvMLPEncoder(
            node_feat_dim=cfg.geo.node_feature_dim,
            n_nodes=node_feats.shape[0],
            embed_dim=cfg.model.embed_dim,
        )
        if model_initializer is not None:
            model = model_initializer(model)
        trainer = Trainer(model, cfg, cfg.train.device, is_multimodal=True)
        histories.append(trainer.train(train_loader, val_loader, fold_id=fold_id))

        acc, f1, auc, _, _ = trainer.evaluate(test_loader)
        fold_metrics["accuracy"].append(acc)
        fold_metrics["macro_f1"].append(f1)
        fold_metrics["roc_auc"].append(auc)

    _log_cv_summary(fold_metrics)
    return fold_metrics, histories


def run_shuffled_labels_ablation(
    X: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    cfg,
    model_initializer: Optional[Callable] = None,
) -> Tuple[Dict[str, List[float]], List[Dict]]:
    from models.fusion import NeuroGeospatialModel

    node_feats, adj = _build_astana_graph_for_ablation(cfg)
    use_cuda = cfg.train.device.startswith("cuda")
    fold_metrics = {"accuracy": [], "macro_f1": [], "roc_auc": []}
    histories = []
    rng = np.random.default_rng(cfg.train.random_seed)

    for fold_id, (train_val_idx, test_idx) in enumerate(
        GroupKFold(cfg.train.n_folds).split(X, labels, groups=subject_ids)
    ):
        tv_subjects = np.unique(subject_ids[train_val_idx])
        val_subjects = tv_subjects[-5:]
        train_idx = train_val_idx[np.isin(subject_ids[train_val_idx], tv_subjects[:-5])]
        val_idx = train_val_idx[np.isin(subject_ids[train_val_idx], val_subjects)]

        pecm = PECM(node_feats, random_state=cfg.train.random_seed)
        env_tr = pecm.sample(rng.permutation(labels[train_idx]))
        env_va = pecm.sample_unconditional(len(val_idx))
        env_te = pecm.sample_unconditional(len(test_idx))

        bs = cfg.train.batch_size
        train_loader = DataLoader(
            MultimodalDataset(X[train_idx], env_tr, node_feats, adj, labels[train_idx]),
            batch_size=bs, shuffle=True, num_workers=0, pin_memory=use_cuda,
        )
        val_loader = DataLoader(
            MultimodalDataset(X[val_idx], env_va, node_feats, adj, labels[val_idx]),
            batch_size=bs, shuffle=False, num_workers=0,
        )
        test_loader = DataLoader(
            MultimodalDataset(X[test_idx], env_te, node_feats, adj, labels[test_idx]),
            batch_size=bs, shuffle=False, num_workers=0,
        )

        model = NeuroGeospatialModel(cfg, bidirectional=True)
        if model_initializer is not None:
            model = model_initializer(model)
        trainer = Trainer(model, cfg, cfg.train.device, is_multimodal=True)
        histories.append(trainer.train(train_loader, val_loader, fold_id=fold_id))

        acc, f1, auc, _, _ = trainer.evaluate(test_loader)
        fold_metrics["accuracy"].append(acc)
        fold_metrics["macro_f1"].append(f1)
        fold_metrics["roc_auc"].append(auc)

    _log_cv_summary(fold_metrics)
    return fold_metrics, histories


def run_gcn_only_ablation(
    X: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    cfg,
    model_initializer: Optional[Callable] = None,
) -> Tuple[Dict[str, List[float]], List[Dict]]:
    from models.fusion import NeuroGeospatialModel

    def _build_model(node_feats, adj):
        model = NeuroGeospatialModel(cfg, bidirectional=True)
        model.env_encoder = _make_stgnn(cfg, use_temporal=False)
        return model

    return _run_multimodal_ablation(
        X=X,
        labels=labels,
        subject_ids=subject_ids,
        cfg=cfg,
        build_model=_build_model,
        train_env_sampler=lambda pecm, y, idx, fold_id: pecm.sample(y),
        model_initializer=model_initializer,
    )


def run_reversed_dose_response_ablation(
    X: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    cfg,
    model_initializer: Optional[Callable] = None,
) -> Tuple[Dict[str, List[float]], List[Dict]]:
    from models.fusion import NeuroGeospatialModel

    return _run_multimodal_ablation(
        X=X,
        labels=labels,
        subject_ids=subject_ids,
        cfg=cfg,
        build_model=lambda node_feats, adj: NeuroGeospatialModel(cfg, bidirectional=True),
        train_env_sampler=lambda pecm, y, idx, fold_id: pecm.sample(y),
        model_initializer=model_initializer,
        pecm_factory=lambda feats: PECM(
            feats,
            random_state=cfg.train.random_seed,
            reverse=True,
        ),
    )


def run_concat_fusion_ablation(
    X: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    cfg,
    model_initializer: Optional[Callable] = None,
) -> Tuple[Dict[str, List[float]], List[Dict]]:
    from models.fusion import NeuroGeospatialModel

    return _run_multimodal_ablation(
        X=X,
        labels=labels,
        subject_ids=subject_ids,
        cfg=cfg,
        build_model=lambda node_feats, adj: NeuroGeospatialModel(
            cfg, bidirectional=True, fusion_type="concat"
        ),
        train_env_sampler=lambda pecm, y, idx, fold_id: pecm.sample(y),
        model_initializer=model_initializer,
    )


def run_infonce_alignment_ablation(
    X: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    cfg,
    model_initializer: Optional[Callable] = None,
    temperature: float = 0.1,
) -> Tuple[Dict[str, List[float]], List[Dict]]:
    from models.fusion import NeuroGeospatialModel

    return _run_multimodal_ablation(
        X=X,
        labels=labels,
        subject_ids=subject_ids,
        cfg=cfg,
        build_model=lambda node_feats, adj: NeuroGeospatialModel(cfg, bidirectional=True),
        train_env_sampler=lambda pecm, y, idx, fold_id: pecm.sample(y),
        model_initializer=model_initializer,
        trainer_kwargs={
            "loss_name": "compute_loss_infonce",
            "loss_kwargs": {"temperature": temperature},
        },
    )


def run_rma_projection_ablation(
    X: np.ndarray,
    epochs_raw: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    cfg,
    model_initializer: Optional[Callable] = None,
) -> Tuple[Dict[str, List[float]], List[Dict]]:
    from data.alignment import RMA, compute_eeg_covariances
    from models.fusion import NeuroGeospatialModel

    if epochs_raw is None:
        raise ValueError("epochs_raw is required for the RMA-projected ablation.")
    if len(epochs_raw) != len(X):
        raise ValueError(
            "epochs_raw and X must have the same number of epochs for RMA alignment."
        )

    node_feats, adj = _build_astana_graph_for_ablation(cfg)
    use_cuda = cfg.train.device.startswith("cuda")
    fold_metrics = {"accuracy": [], "macro_f1": [], "roc_auc": []}
    histories = []

    for fold_id, train_idx, val_idx, test_idx in _iter_subject_folds(
        X, labels, subject_ids, cfg
    ):
        pecm = PECM(node_feats, random_state=cfg.train.random_seed)
        env_tr = pecm.sample(labels[train_idx])
        env_va = pecm.sample_unconditional(len(val_idx))
        env_te = pecm.sample_unconditional(len(test_idx))

        logger.info(
            "FOLD %d: fitting frozen RMA projection on %d training epochs.",
            fold_id + 1,
            len(train_idx),
        )
        rma = RMA(embed_dim=cfg.model.embed_dim)
        eeg_cov_tr = compute_eeg_covariances(epochs_raw[train_idx])
        _, rma_env_tr = rma.fit(eeg_cov_tr, env_tr)

        bs = cfg.train.batch_size
        train_loader = DataLoader(
            MultimodalAlignmentDataset(
                X[train_idx],
                env_tr,
                node_feats,
                adj,
                rma_env_tr,
                labels[train_idx],
            ),
            batch_size=bs,
            shuffle=True,
            num_workers=0,
            pin_memory=use_cuda,
        )
        val_loader = DataLoader(
            MultimodalDataset(X[val_idx], env_va, node_feats, adj, labels[val_idx]),
            batch_size=bs,
            shuffle=False,
            num_workers=0,
        )
        test_loader = DataLoader(
            MultimodalDataset(X[test_idx], env_te, node_feats, adj, labels[test_idx]),
            batch_size=bs,
            shuffle=False,
            num_workers=0,
        )

        model = NeuroGeospatialModel(cfg, bidirectional=True)
        if model_initializer is not None:
            model = model_initializer(model)
        trainer = Trainer(model, cfg, cfg.train.device, is_multimodal=True)
        histories.append(trainer.train(train_loader, val_loader, fold_id=fold_id))

        acc, f1, auc, _, _ = trainer.evaluate(test_loader)
        fold_metrics["accuracy"].append(acc)
        fold_metrics["macro_f1"].append(f1)
        fold_metrics["roc_auc"].append(auc)

    _log_cv_summary(fold_metrics)
    return fold_metrics, histories


def run_transfer_experiment(
    X:                     np.ndarray,
    labels:                np.ndarray,
    subject_ids:           np.ndarray,
    cfg,
    pretrained_model_state: dict,
) -> Tuple[Dict[str, List[float]], List[Dict]]:
    from models.fusion import NeuroGeospatialModel

    def _transfer_factory():
        model = NeuroGeospatialModel(cfg, bidirectional=True)
        model.load_state_dict(pretrained_model_state, strict=False)

        for param in model.eeg_encoder.parameters():
            param.requires_grad = False

        model.env_encoder = _make_stgnn(cfg, use_temporal=True)
        eeg_params = sum(p.numel() for p in model.eeg_encoder.parameters())
        env_params = sum(p.numel() for p in model.env_encoder.parameters())
        logger.info("Env-swap model: EEG-Conformer frozen (%d params), "
                    "ST-GNN re-initialised (%d params).", eeg_params, env_params)
        return model

    logger.info("Running Singapore environmental-domain swap ...")
    return run_subject_cv(
        X=X, labels=labels, subject_ids=subject_ids, cfg=cfg,
        model_factory=_transfer_factory,
        mode="multimodal",
        city="singapore",
    )


def _log_cv_summary(fold_metrics: Dict[str, List[float]]):
    logger.info("CV SUMMARY")
    for metric, vals in fold_metrics.items():
        arr = np.array(vals)
        logger.info("  %-12s %.4f +/- %.4f", metric, arr.mean(), arr.std())
