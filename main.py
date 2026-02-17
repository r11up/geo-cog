#!/usr/bin/env python3
"""Run the EAV neuro-geospatial experiments, optional SEED-IV initialization, ablations, interpretation, and MAUP checks."""

import argparse
import csv
import logging
import random
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["all", "baselines", "proposed", "ablation", "maup", "transfer"],
        default="all",
    )
    p.add_argument("--data_root",  type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     type=str, default=None)
    p.add_argument("--seediv_ckpt", type=str, default=None,
                   help="Path to EEG-Conformer weights pretrained on SEED-IV")
    p.add_argument("--gee_project",type=str, default=None,
                   help="Google Earth Engine cloud project ID")
    p.add_argument("--openaq_key", type=str, default=None,
                   help="OpenAQ v3 API key (recommended; may be required)")
    p.add_argument("--year",       type=int, default=2025,
                   help="Year for satellite composites")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _summarize_fold_metrics(metrics):
    return {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for k, v in metrics.items()
    }


def _format_accuracy(summary):
    return f"{summary['accuracy']['mean'] * 100:.1f} ± {summary['accuracy']['std'] * 100:.1f}"


def _write_oracle_comparison_csv(
    output_dir: Path,
    eeg_only_summary,
    realistic_summary,
    oracle_summary,
):
    rows = []
    if eeg_only_summary is not None:
        rows.append({
            "Test-time environmental conditioning": "EEG-only",
            "Accuracy": _format_accuracy(eeg_only_summary),
        })
    rows.extend([
        {
            "Test-time environmental conditioning": "Unconditional (realistic inference)",
            "Accuracy": _format_accuracy(realistic_summary),
        },
        {
            "Test-time environmental conditioning": "Oracle / label-conditioned at test",
            "Accuracy": _format_accuracy(oracle_summary),
        },
    ])

    path = output_dir / "table3b_oracle_vs_realistic.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Test-time environmental conditioning", "Accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Oracle comparison CSV saved to %s", path)


def _load_seediv_weights(model, ckpt_path: Path, device: str):
    import torch

    if not ckpt_path or not ckpt_path.exists():
        cache_key = str(ckpt_path)
        warned = getattr(_load_seediv_weights, "_warned_missing", set())
        if cache_key not in warned:
            logger.warning(
                "SEED-IV checkpoint not found at %s; using random EEG encoder weights.",
                ckpt_path,
            )
            warned.add(cache_key)
            _load_seediv_weights._warned_missing = warned
        return model

    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    target_prefix = None
    if hasattr(model, "eeg_encoder"):
        target_prefix = "eeg_encoder."
    elif hasattr(model, "encoder"):
        target_prefix = "encoder."
    if target_prefix is None:
        logger.warning("SEED-IV weights skipped; model has no EEG encoder.")
        return model

    model_state = model.state_dict()
    updates = {}
    for key, value in state.items():
        stripped = key
        for prefix in ("eeg_encoder.", "encoder."):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
                break
        target_key = target_prefix + stripped
        if target_key in model_state and model_state[target_key].shape == value.shape:
            updates[target_key] = value

    model_state.update(updates)
    model.load_state_dict(model_state)
    logger.info("Loaded %d SEED-IV EEG encoder tensors from %s",
                len(updates), ckpt_path)
    return model


def main():
    args = parse_args()
    from config import CFG

    if args.data_root:
        CFG.paths.eav_root = Path(args.data_root)
    if args.output_dir:
        CFG.paths.output_dir = Path(args.output_dir)
        CFG.paths.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device:
        CFG.train.device = args.device
    if args.seediv_ckpt:
        CFG.paths.seediv_pretrain_ckpt = Path(args.seediv_ckpt)

    CFG.train.random_seed = args.seed
    CFG.geo.gee_project    = getattr(args, "gee_project", None)
    CFG.geo.openaq_api_key = getattr(args, "openaq_key", None)
    CFG.geo.satellite_year  = getattr(args, "year", 2025)
    set_seed(args.seed)

    logger.info("EAV root  : %s", CFG.paths.eav_root)
    logger.info("Device    : %s", CFG.train.device)
    logger.info("SEED-IV   : %s", CFG.paths.seediv_pretrain_ckpt)
    logger.info("Primary   : Astana  (EPSG:%d)", CFG.geo.epsg_astana)
    logger.info("Env swap  : Singapore (EPSG:%d)", CFG.geo.epsg_singapore)

    from data.eeg_preprocessing import preprocess_all
    epochs_raw, labels, subject_ids = preprocess_all(CFG)
    logger.info("Epochs (raw): %s", epochs_raw.shape)

    from data.eeg_features import extract_features_cached
    X = extract_features_cached(epochs_raw, CFG)
    logger.info("Features X: %s", X.shape)

    from training.cross_validation import run_subject_cv, run_transfer_experiment
    from evaluation.metrics import print_results_table
    from evaluation.visualization import (
        plot_graphlime_environmental_attribution_figure9,
        plot_results_table,
        plot_training_curves,
    )

    all_results = {}
    proposed_model_state = None

    if args.mode in ("all", "baselines"):
        from models.baselines import RandomForestBaseline
        from sklearn.model_selection import GroupKFold
        from evaluation.metrics import compute_metrics

        logger.info("BASELINE — Random Forest (EEG-only)")
        rf_m = {"accuracy": [], "macro_f1": [], "roc_auc": []}
        for _, (tr, te) in enumerate(
            GroupKFold(CFG.train.n_folds).split(X, labels, groups=subject_ids)
        ):
            rf = RandomForestBaseline(random_state=CFG.train.random_seed)
            rf.fit(X[tr], labels[tr])
            m = compute_metrics(labels[te], rf.predict(X[te]),
                                rf.predict_proba(X[te]))
            rf_m["accuracy"].append(m.accuracy)
            rf_m["macro_f1"].append(m.macro_f1)
            rf_m["roc_auc"].append(m.roc_auc)
        all_results["Random Forest (EEG-only)"] = {
            k: {"mean": np.mean(v), "std": np.std(v)} for k, v in rf_m.items()
        }

    if args.mode in ("all", "baselines"):
        from models.baselines import BidirectionalLSTM
        logger.info("BASELINE — Bidirectional LSTM (EEG-only)")
        lstm_m, lstm_h = run_subject_cv(
            X, labels, subject_ids, CFG,
            model_factory=lambda: BidirectionalLSTM(
                input_dim=10, hidden_dim=256, n_layers=2,
                n_classes=CFG.eav.n_classes),
            mode="eeg_only", city="astana",
        )
        all_results["LSTM (EEG-only)"] = {
            k: {"mean": np.mean(v), "std": np.std(v)} for k, v in lstm_m.items()
        }
        plot_training_curves(lstm_h, CFG.paths.output_dir, "lstm_training")

    if args.mode in ("all", "baselines"):
        from models.baselines import EEGConformerOnly
        logger.info("BASELINE — EEG-Conformer (EEG-only)")
        def _conformer_factory():
            return _load_seediv_weights(
                EEGConformerOnly(CFG),
                CFG.paths.seediv_pretrain_ckpt,
                CFG.train.device,
            )
        conf_m, conf_h = run_subject_cv(
            X, labels, subject_ids, CFG,
            model_factory=_conformer_factory,
            mode="eeg_only", city="astana",
        )
        all_results["EEG-Conformer (EEG-only)"] = {
            k: {"mean": np.mean(v), "std": np.std(v)} for k, v in conf_m.items()
        }
        plot_training_curves(conf_h, CFG.paths.output_dir, "conformer_training")

    if args.mode in ("all", "baselines"):
        from models.baselines import EnvironmentOnlyMLP
        logger.info("BASELINE — Environment-only MLP (Astana)")
        env_m, _ = run_subject_cv(
            X, labels, subject_ids, CFG,
            model_factory=lambda: EnvironmentOnlyMLP(
                node_feat_dim=CFG.geo.node_feature_dim,
                n_classes=CFG.eav.n_classes),
            mode="env_only", city="astana",
        )
        all_results["Environment-only (Astana)"] = {
            k: {"mean": np.mean(v), "std": np.std(v)} for k, v in env_m.items()
        }

    if args.mode in ("all", "proposed", "ablation"):
        from models.fusion import NeuroGeospatialModel
        logger.info("ABLATION — Unidirectional attention (Astana)")
        def _uni_factory():
            return _load_seediv_weights(
                NeuroGeospatialModel(CFG, bidirectional=False),
                CFG.paths.seediv_pretrain_ckpt,
                CFG.train.device,
            )
        uni_m, _ = run_subject_cv(
            X, labels, subject_ids, CFG,
            model_factory=_uni_factory,
            mode="multimodal", city="astana",
        )
        all_results["EEG-Conformer + ST-GNN (unidirectional)"] = {
            k: {"mean": np.mean(v), "std": np.std(v)} for k, v in uni_m.items()
        }

    if args.mode in ("all", "ablation"):
        from training.cross_validation import (
            run_concat_fusion_ablation,
            run_env_mlp_ablation,
            run_gcn_only_ablation,
            run_infonce_alignment_ablation,
            run_no_alignment_ablation,
            run_reversed_dose_response_ablation,
            run_rma_projection_ablation,
            run_shuffled_labels_ablation,
        )
        from models.baselines import EEGNetBaseline

        def _init_seediv(model):
            return _load_seediv_weights(
                model, CFG.paths.seediv_pretrain_ckpt, CFG.train.device
            )

        logger.info("ABLATION — No alignment (random env pairing)")
        no_align_m, _ = run_no_alignment_ablation(
            X, labels, subject_ids, CFG, model_initializer=_init_seediv,
        )
        all_results["No alignment (random env pairing)"] = {
            k: {"mean": np.mean(v), "std": np.std(v)} for k, v in no_align_m.items()
        }

        logger.info("ABLATION — Shuffled PECM labels")
        shuffled_m, _ = run_shuffled_labels_ablation(
            X, labels, subject_ids, CFG, model_initializer=_init_seediv,
        )
        all_results["Shuffled env labels (PECM disabled)"] = {
            k: {"mean": np.mean(v), "std": np.std(v)} for k, v in shuffled_m.items()
        }

        logger.info("ABLATION — GCN-only, no temporal module")
        gcn_only_m, _ = run_gcn_only_ablation(
            X, labels, subject_ids, CFG, model_initializer=_init_seediv,
        )
        all_results["GCN-only, no temporal module"] = {
            k: {"mean": np.mean(v), "std": np.std(v)} for k, v in gcn_only_m.items()
        }

        logger.info("ABLATION — Reversed dose-response")
        reversed_m, _ = run_reversed_dose_response_ablation(
            X, labels, subject_ids, CFG, model_initializer=_init_seediv,
        )
        all_results["Reversed dose-response (negative control)"] = {
            k: {"mean": np.mean(v), "std": np.std(v)} for k, v in reversed_m.items()
        }

        logger.info("ABLATION — Concatenation fusion")
        concat_m, _ = run_concat_fusion_ablation(
            X, labels, subject_ids, CFG, model_initializer=_init_seediv,
        )
        all_results["Concatenation fusion (no cross-modal attention)"] = {
            k: {"mean": np.mean(v), "std": np.std(v)} for k, v in concat_m.items()
        }

        logger.info("ABLATION — Contrastive InfoNCE alignment")
        infonce_m, _ = run_infonce_alignment_ablation(
            X, labels, subject_ids, CFG, model_initializer=_init_seediv,
        )
        all_results["Contrastive InfoNCE alignment"] = {
            k: {"mean": np.mean(v), "std": np.std(v)} for k, v in infonce_m.items()
        }

        logger.info("ABLATION — Frozen RMA-projected alignment")
        rma_m, _ = run_rma_projection_ablation(
            X, epochs_raw, labels, subject_ids, CFG, model_initializer=_init_seediv,
        )
        all_results["Frozen RMA-projected alignment"] = {
            k: {"mean": np.mean(v), "std": np.std(v)} for k, v in rma_m.items()
        }

        logger.info("ABLATION — EEG-Conformer + MLP-env (no graph)")
        env_mlp_m, _ = run_env_mlp_ablation(
            X, labels, subject_ids, CFG, model_initializer=_init_seediv,
        )
        all_results["EEG-Conformer + MLP-env (no graph)"] = {
            k: {"mean": np.mean(v), "std": np.std(v)} for k, v in env_mlp_m.items()
        }

        logger.info("ABLATION — CNN backbone (EEGNet)")
        eegnet_m, _ = run_subject_cv(
            X, labels, subject_ids, CFG,
            model_factory=lambda: EEGNetBaseline(CFG),
            mode="eeg_only", city="astana",
        )
        all_results["CNN backbone (EEGNet, no Conformer)"] = {
            k: {"mean": np.mean(v), "std": np.std(v)} for k, v in eegnet_m.items()
        }

    if args.mode in ("all", "proposed"):
        from models.fusion import NeuroGeospatialModel
        from evaluation.attribution import GraphLIMEAttributor
        import torch
        logger.info("PROPOSED — Bidirectional [PRIMARY: Astana]")

        _ref = [None]
        graphlime_fold_attributions = []

        def _bidir_factory():
            m = _load_seediv_weights(
                NeuroGeospatialModel(CFG, bidirectional=True),
                CFG.paths.seediv_pretrain_ckpt,
                CFG.train.device,
            )
            _ref[0] = m
            return m

        def _collect_graphlime_fold_attribution(**fold_context):
            X_test = fold_context["X_test"]
            logits = fold_context["logits"]
            n_interp = min(100, len(X_test))
            if n_interp == 0:
                return

            predicted_classes = np.asarray(logits).argmax(axis=1)[:n_interp]
            attributor = GraphLIMEAttributor(
                fold_context["model"],
                CFG.train.device,
                CFG.eav.n_classes,
            )
            attribution = attributor.attribute(
                X_test[:n_interp],
                fold_context["node_feats"],
                fold_context["adj"],
                predicted_classes,
                n_perturb=50,
                seed=CFG.train.random_seed + fold_context["fold_id"],
            )
            graphlime_fold_attributions.append(attribution)

        bd_m, bd_h = run_subject_cv(
            X, labels, subject_ids, CFG,
            model_factory=_bidir_factory,
            mode="multimodal", city="astana",
            fold_callback=_collect_graphlime_fold_attribution,
        )
        proposed_summary = _summarize_fold_metrics(bd_m)
        all_results["EEG-Conformer + ST-GNN (bidir., proposed)"] = proposed_summary
        plot_training_curves(bd_h, CFG.paths.output_dir, "proposed_training")
        if len(graphlime_fold_attributions) != CFG.train.n_folds:
            raise RuntimeError(
                "Expected GraphLIME attributions for "
                f"{CFG.train.n_folds} folds, got {len(graphlime_fold_attributions)}."
            )
        graphlime_arr = np.stack(graphlime_fold_attributions, axis=0)
        graphlime_path = CFG.paths.output_dir / "figure9_graphlime_fold_attributions.npy"
        np.save(graphlime_path, graphlime_arr)
        logger.info("Saved Figure 9 fold GraphLIME attributions -> %s", graphlime_path)
        plot_graphlime_environmental_attribution_figure9(
            CFG.paths.output_dir,
            fold_attributions=graphlime_arr,
            class_names=list(CFG.eav.class_names),
        )

        if _ref[0] is not None:
            proposed_model_state = _ref[0].state_dict()
            ckpt = CFG.paths.checkpoint_dir / "proposed_astana_best.pt"
            torch.save(proposed_model_state, str(ckpt))
            logger.info("Saved Astana model -> %s", ckpt)

        logger.info("PROPOSED — Bidirectional oracle PECM test [upper bound]")

        def _oracle_factory():
            return _load_seediv_weights(
                NeuroGeospatialModel(CFG, bidirectional=True),
                CFG.paths.seediv_pretrain_ckpt,
                CFG.train.device,
            )

        oracle_m, _ = run_subject_cv(
            X, labels, subject_ids, CFG,
            model_factory=_oracle_factory,
            mode="multimodal_oracle_test", city="astana",
        )
        oracle_summary = _summarize_fold_metrics(oracle_m)
        _write_oracle_comparison_csv(
            CFG.paths.output_dir,
            all_results.get("EEG-Conformer (EEG-only)"),
            proposed_summary,
            oracle_summary,
        )

    if all_results:
        logger.info("\n--- PRIMARY RESULTS (Astana) ---")
        print_results_table(all_results)
        plot_results_table(all_results, CFG.paths.output_dir,
                           filename="table3_astana_results")

    if args.mode in ("all", "transfer"):
        import torch
        logger.info("SINGAPORE ENVIRONMENTAL-DOMAIN SWAP")

        ckpt = CFG.paths.checkpoint_dir / "proposed_astana_best.pt"
        if proposed_model_state is None:
            if ckpt.exists():
                proposed_model_state = torch.load(str(ckpt), map_location="cpu",
                                                  weights_only=True)

        if proposed_model_state is not None:
            sg_m, _ = run_transfer_experiment(
                X=X, labels=labels, subject_ids=subject_ids, cfg=CFG,
                pretrained_model_state=proposed_model_state,
            )
            all_results["EEG-Conformer + ST-GNN (bidir., Singapore env swap)"] = {
                k: {"mean": np.mean(v), "std": np.std(v)} for k, v in sg_m.items()
            }

            from models.baselines import EnvironmentOnlyMLP
            logger.info("BASELINE — Environment-only MLP (Singapore env swap)")
            sg_env_m, _ = run_subject_cv(
                X, labels, subject_ids, CFG,
                model_factory=lambda: EnvironmentOnlyMLP(
                    node_feat_dim=CFG.geo.node_feature_dim,
                    n_classes=CFG.eav.n_classes),
                mode="env_only", city="singapore",
            )
            all_results["Environment-only (Singapore env swap)"] = {
                k: {"mean": np.mean(v), "std": np.std(v)} for k, v in sg_env_m.items()
            }

            logger.info("\n--- SINGAPORE ENV-SWAP RESULTS ---")
            print_results_table({k: v for k, v in all_results.items()
                                  if "Singapore" in k})
            plot_results_table(all_results, CFG.paths.output_dir,
                               filename="table4_transfer_results")
        else:
            logger.warning("Skipping Singapore env swap; no Astana checkpoint found at %s", ckpt)

    if args.mode in ("all", "proposed"):
        _run_interpretation(X, epochs_raw, labels, subject_ids, CFG)

    if args.mode in ("all", "maup"):
        _run_maup_analysis(X, epochs_raw, labels, subject_ids, CFG)

    logger.info("Done. Outputs saved to %s", CFG.paths.output_dir)


def _run_interpretation(X, epochs_raw, labels, subject_ids, CFG):
    import torch
    from sklearn.model_selection import GroupKFold
    from data.geospatial import build_astana_grid
    from data.alignment import PECM
    from models.fusion import NeuroGeospatialModel
    from training.trainer import Trainer
    from training.cross_validation import MultimodalDataset
    from evaluation.attribution import GraphLIMEAttributor
    from evaluation.visualization import (
        plot_spatial_error_map, compute_spatial_error_rates,
        plot_attribution, plot_confusion_matrix,
    )
    from evaluation.metrics import compute_metrics
    from torch.utils.data import DataLoader

    node_feats, adj = build_astana_grid(CFG)
    adj_t = torch.from_numpy(adj).float()

    splits = list(GroupKFold(CFG.train.n_folds).split(X, labels, groups=subject_ids))
    train_val_idx, test_idx = splits[0]
    tv_subj    = np.unique(subject_ids[train_val_idx])
    val_subj   = tv_subj[-5:]
    train_subj = tv_subj[:-5]
    train_idx = train_val_idx[np.isin(subject_ids[train_val_idx], train_subj)]
    val_idx   = train_val_idx[np.isin(subject_ids[train_val_idx], val_subj)]

    pecm    = PECM(node_feats, random_state=CFG.train.random_seed)
    logger.info("Sampling label-conditioned training PECM contexts for interpretation.")
    y_train = labels[train_idx]; env_train = pecm.sample(y_train)
    y_val   = labels[val_idx];   env_val   = pecm.sample_unconditional(len(y_val))
    y_test  = labels[test_idx];  env_test  = pecm.sample_unconditional(len(y_test))

    bs = CFG.train.batch_size
    train_loader = DataLoader(
        MultimodalDataset(X[train_idx], env_train, node_feats, adj, y_train),
        batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        MultimodalDataset(X[val_idx], env_val, node_feats, adj, y_val),
        batch_size=bs, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        MultimodalDataset(X[test_idx], env_test, node_feats, adj, y_test),
        batch_size=bs, shuffle=False, num_workers=0)

    model   = _load_seediv_weights(
        NeuroGeospatialModel(CFG, bidirectional=True),
        CFG.paths.seediv_pretrain_ckpt,
        CFG.train.device,
    )
    trainer = Trainer(model, CFG, CFG.train.device, is_multimodal=True)
    trainer.train(train_loader, val_loader, fold_id=99)

    acc, f1, auc, logits, targets = trainer.evaluate(test_loader)
    preds = logits.argmax(axis=1)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    m = compute_metrics(targets, preds, probs)
    logger.info("Interpretation (Astana): acc=%.4f  f1=%.4f  auc=%.4f", acc, f1, auc)

    plot_confusion_matrix(m.confusion_mat, CFG.paths.output_dir,
                          title="Proposed Model (Astana)")

    n_interp   = min(100, len(test_idx))
    attributor = GraphLIMEAttributor(model, CFG.train.device, CFG.eav.n_classes)
    attribution = attributor.attribute(
        X[test_idx[:n_interp]], node_feats, adj,
        labels[test_idx[:n_interp]], n_perturb=50,
    )
    for cls, ranked in attributor.rank_features(
        attribution, class_names=CFG.eav.class_names
    ).items():
        logger.info("Attribution [%s]: %s", cls, ranked[:3])
    plot_attribution(attribution, CFG.paths.output_dir,
                     class_names=list(CFG.eav.class_names))

    err_rates = compute_spatial_error_rates(
        targets, preds, n_nodes=CFG.geo.n_nodes, n_classes=CFG.eav.n_classes)
    logger.warning(
        "EAV has no per-epoch GPS/node IDs; spatial error map uses proxy assignment."
    )
    plot_spatial_error_map(err_rates, CFG.paths.output_dir)

    model.eval()
    all_eeg, all_env = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_eeg, x_env, _adj, _ = [b.to(CFG.train.device) for b in batch]
            _, v_eeg, v_env = model(x_eeg, x_env, adj_t.to(CFG.train.device))
            all_eeg.append(v_eeg.cpu().numpy())
            all_env.append(v_env.cpu().numpy())
    from evaluation.visualization import plot_latent_space
    plot_latent_space(
        np.concatenate(all_eeg), np.concatenate(all_env),
        targets, CFG.paths.output_dir, stage="trained_neural_alignment")


def _run_maup_analysis(X, epochs_raw, labels, subject_ids, CFG):
    from data.geospatial import build_astana_grid, coarsen_pollution_features
    from models.fusion import NeuroGeospatialModel
    from training.cross_validation import run_subject_cv
    from evaluation.metrics import print_results_table
    from evaluation.visualization import plot_results_table
    import data.geospatial as _geo

    node_feats_native, adj = build_astana_grid(CFG)
    maup_results = {}

    for km in CFG.geo.maup_resolutions:
        logger.info("MAUP — %d km2 (Astana)", km)
        _nf = (node_feats_native if km == 1
               else coarsen_pollution_features(node_feats_native, CFG.geo.grid_size, km))

        _orig = _geo.build_astana_grid
        _geo.build_astana_grid = lambda c: (_nf, adj)
        try:
            m, _ = run_subject_cv(
                X, labels, subject_ids, CFG,
                model_factory=lambda: _load_seediv_weights(
                    NeuroGeospatialModel(CFG, bidirectional=True),
                    CFG.paths.seediv_pretrain_ckpt,
                    CFG.train.device,
                ),
                mode="multimodal", city="astana",
            )
        finally:
            _geo.build_astana_grid = _orig

        maup_results[f"{km} km2"] = {
            k: {"mean": np.mean(v), "std": np.std(v)} for k, v in m.items()
        }

    print_results_table(maup_results)
    plot_results_table(maup_results, CFG.paths.output_dir, filename="table5_maup")


if __name__ == "__main__":
    main()
