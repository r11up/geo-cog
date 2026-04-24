"""Plot result tables, curves, confusion matrices, latent spaces, spatial maps, and attribution figures."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

logger = logging.getLogger(__name__)

CLASS_NAMES = ["neutral", "sadness", "anger", "happiness", "relaxed"]
ENV_FEAT_NAMES = ["PM2.5", "NDVI", "FAR", "GVI", "NO2", "O3"]


def _display_class_names(class_names: Optional[List[str]] = None) -> List[str]:
    if class_names is None:
        class_names = CLASS_NAMES
    display = []
    for name in class_names:
        clean = str(name).replace("_", " ").strip()
        if clean.lower() in {"relaxed", "relaxed/calm", "relaxed calm"}:
            display.append("Relaxed-Calm")
        else:
            display.append(clean.title())
    return display


def _save(fig, path: Path, name: str):
    out_pdf = path / f"{name}.pdf"
    out_png = path / f"{name}.png"
    fig.savefig(str(out_pdf), bbox_inches="tight", dpi=300)
    fig.savefig(str(out_png), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_png)


def plot_results_table(
    results: Dict[str, Dict],
    output_dir: Path,
    filename: str = "table2_results",
):
    import pandas as pd
    rows = []
    for model_name, metrics in results.items():
        acc_m = metrics["accuracy"]["mean"] * 100
        acc_s = metrics["accuracy"]["std"]  * 100
        f1_m  = metrics["macro_f1"]["mean"]
        f1_s  = metrics["macro_f1"]["std"]
        auc_m = metrics["roc_auc"]["mean"]
        auc_s = metrics["roc_auc"]["std"]
        rows.append({
            "Model":        model_name,
            "Accuracy (%)": f"{acc_m:.1f} ± {acc_s:.1f}",
            "Macro F1":     f"{f1_m:.3f} ± {f1_s:.3f}",
            "ROC-AUC":      f"{auc_m:.3f} ± {auc_s:.3f}",
        })
    df = pd.DataFrame(rows)
    csv_path = output_dir / f"{filename}.csv"
    df.to_csv(str(csv_path), index=False)
    logger.info("Results CSV saved to %s", csv_path)

    fig, ax = plt.subplots(figsize=(12, max(3, len(rows) * 0.6 + 1)))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    ax.set_title("Table 2 — Five-fold Cross-validated Mental State Classification",
                 pad=10, fontsize=11, fontweight="bold")
    _save(fig, output_dir, filename)


def plot_training_curves(
    histories: List[Dict],
    output_dir: Path,
    filename: str = "training_curves",
):
    n_folds = len(histories)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for fold_id, h in enumerate(histories):
        epochs = range(1, len(h["train_acc"]) + 1)
        axes[0].plot(epochs, h["val_acc"],   label=f"Fold {fold_id+1} val_acc", alpha=0.7)
        axes[1].plot(epochs, h["val_f1"],    label=f"Fold {fold_id+1} val_f1",  alpha=0.7)

    axes[0].set_title("Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=7)

    axes[1].set_title("Validation Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    _save(fig, output_dir, filename)


def plot_confusion_matrix(
    confusion_mat: np.ndarray,
    output_dir: Path,
    class_names: List[str] = CLASS_NAMES,
    filename: str = "confusion_matrix",
    title: str = "Confusion Matrix",
):
    cm_norm = confusion_mat.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm_norm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0.0,
        vmax=1.0,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, output_dir, filename)


def plot_latent_space(
    z_eeg: np.ndarray,
    z_env: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    stage: str = "trained neural alignment",
    class_names: List[str] = CLASS_NAMES,
    filename: Optional[str] = None,
    method: str = "auto",
):
    try:
        import umap
        _has_umap = True
    except ImportError:
        _has_umap = False

    Z = np.concatenate([z_eeg, z_env], axis=0)
    N = len(z_eeg)

    use_umap = (method == "umap") or (method == "auto" and _has_umap)

    if use_umap:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
        Z2d = reducer.fit_transform(Z)
        method_label = "UMAP"
    else:
        from sklearn.manifold import TSNE
        Z2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(Z)
        method_label = "t-SNE"

    z_eeg_2d = Z2d[:N]
    z_env_2d = Z2d[N:]

    colours = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    fig, ax = plt.subplots(figsize=(8, 6))
    for c, (cls_name, col) in enumerate(zip(class_names, colours)):
        mask = labels == c
        ax.scatter(z_eeg_2d[mask, 0], z_eeg_2d[mask, 1],
                   c=[col], marker="o", s=18, alpha=0.6, label=f"{cls_name} (EEG)")
        ax.scatter(z_env_2d[mask, 0], z_env_2d[mask, 1],
                   c=[col], marker="^", s=18, alpha=0.6, label=f"{cls_name} (ENV)")

    stage_label = stage.replace("_", " ").strip().title()
    ax.set_title(f"{method_label} — Shared Latent Space ({stage_label})")
    ax.set_xlabel(f"{method_label} 1")
    ax.set_ylabel(f"{method_label} 2")

    handles, lbls = ax.get_legend_handles_labels()
    eeg_handles = handles[::2]
    eeg_lbls = [l.replace(" (EEG)", "") for l in lbls[::2]]
    ax.legend(eeg_handles, eeg_lbls, fontsize=8, loc="best", title="Class (○ EEG, △ ENV)")

    fig.tight_layout()
    safe_stage = stage.lower().replace(" ", "_")
    fname = filename or f"latent_space_{safe_stage}"
    _save(fig, output_dir, fname)


def plot_spatial_error_map(
    error_rates: np.ndarray,
    output_dir: Path,
    grid_size: int = 10,
    filename: str = "spatial_error_map",
    title: str = "Figure 8 — Proxy Spatial Error Map (No Epoch GPS)",
):
    grid = error_rates.reshape(grid_size, grid_size)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, cmap="Reds", vmin=0.0, vmax=1.0, origin="upper")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Observed / Proxy Error Rate")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Grid Column (West → East)")
    ax.set_ylabel("Grid Row (North → South)")
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))

    for r in range(grid_size):
        for c in range(grid_size):
            val = grid[r, c]
            colour = "white" if val > 0.6 else "black"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=colour)

    fig.tight_layout()
    _save(fig, output_dir, filename)


def compute_spatial_error_rates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_nodes: int = 100,
    n_classes: int = 5,
    node_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    error_rates = np.zeros(n_nodes, dtype=np.float64)
    counts      = np.zeros(n_nodes, dtype=np.float64)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    if node_ids is not None:
        node_ids = np.asarray(node_ids, dtype=np.int64)
        if node_ids.shape[0] != len(y_true):
            raise ValueError("node_ids must have the same length as y_true.")

        valid = (node_ids >= 0) & (node_ids < n_nodes)
        errors = (y_true != y_pred).astype(np.float64)
        for node, is_error in zip(node_ids[valid], errors[valid]):
            counts[node] += 1
            error_rates[node] += is_error

        nonzero = counts > 0
        error_rates[nonzero] /= counts[nonzero]
        return error_rates

    nodes_per_class = n_nodes // n_classes
    class_counters = np.zeros(n_classes, dtype=np.int64)

    for i in range(len(y_true)):
        cls  = int(y_true[i])
        if cls < 0 or cls >= n_classes:
            continue
        base = cls * nodes_per_class
        node = base + (class_counters[cls] % nodes_per_class)
        node = min(node, n_nodes - 1)
        class_counters[cls] += 1

        counts[node]      += 1
        error_rates[node] += int(y_true[i] != y_pred[i])

    nonzero = counts > 0
    error_rates[nonzero] /= counts[nonzero]
    return error_rates


def plot_attribution(
    attribution: np.ndarray,
    output_dir: Path,
    class_names: List[str] = CLASS_NAMES,
    feat_names: List[str] = ENV_FEAT_NAMES,
    filename: str = "feature_attribution",
):
    n_classes, n_feats = attribution.shape
    x = np.arange(n_feats)
    width = 0.8 / n_classes

    fig, ax = plt.subplots(figsize=(10, 5))
    for c in range(n_classes):
        offset = (c - n_classes / 2) * width + width / 2
        ax.bar(x + offset, attribution[c], width,
               label=class_names[c], alpha=0.82)

    ax.set_xticks(x)
    ax.set_xticklabels(feat_names)
    ax.set_ylabel("Mean Absolute Attribution (|β|)")
    ax.set_title("GraphLIME Environmental Feature Attribution by Affective Class")
    ax.legend(title="Class", fontsize=9)
    fig.tight_layout()
    _save(fig, output_dir, filename)


def plot_graphlime_environmental_attribution_figure9(
    output_dir: Path,
    fold_attributions: np.ndarray,
    class_names: Optional[List[str]] = None,
    feat_names: Optional[List[str]] = None,
    filename: str = "figure9_graphlime_environmental_attribution",
):
    """Plot Figure 9: GraphLIME attribution by predicted affective class.

    Parameters
    ----------
    output_dir:
        Directory where PNG/PDF files will be written.
    fold_attributions:
        Real GraphLIME values shaped
        (n_folds, n_predicted_classes, n_features).
    class_names:
        Predicted affective class names, in the same order as the attribution
        array.
    feat_names:
        Equation 9 environmental indicators. Defaults to
        PM2.5, NDVI, FAR, GVI, NO2, O3.
    filename:
        Output stem for both PNG and PDF files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if fold_attributions is None:
        raise ValueError(
            "fold_attributions is required; compute GraphLIME on "
            "cross-validation folds before plotting Figure 9."
        )

    arr = np.asarray(fold_attributions, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(
            "fold_attributions must have shape "
            "(n_folds, n_predicted_classes, n_features)."
        )

    n_folds, n_classes, n_feats = arr.shape
    feat_names = feat_names or ENV_FEAT_NAMES
    class_names = _display_class_names(class_names)
    if len(feat_names) != n_feats:
        raise ValueError(f"Expected {n_feats} feature names, got {len(feat_names)}.")
    if len(class_names) != n_classes:
        raise ValueError(f"Expected {n_classes} class names, got {len(class_names)}.")

    means = arr.mean(axis=0)
    sds = arr.std(axis=0, ddof=1) if n_folds > 1 else np.zeros_like(means)

    feature_colors = {
        "PM2.5": "#c0392b",
        "NDVI": "#2e8b57",
        "FAR":  "#2f5f98",
        "GVI":  "#59a14f",
        "NO2":  "#7b4ab8",
        "O3":   "#2aa1b3",
    }
    colors = [feature_colors.get(name, "#666666") for name in feat_names]

    fig, axes = plt.subplots(
        1, n_classes, figsize=(max(14, 3.2 * n_classes), 5.2), sharey=True
    )
    axes = np.atleast_1d(axes)
    x = np.arange(n_feats)

    for class_idx, ax in enumerate(axes):
        ax.bar(
            x,
            means[class_idx],
            yerr=sds[class_idx],
            color=colors,
            edgecolor="#222222",
            linewidth=0.5,
            capsize=3,
            alpha=0.9,
        )
        ax.set_title(class_names[class_idx], fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(feat_names, rotation=45, ha="right")
        ax.set_ylim(0, max(0.75, float((means + sds).max()) + 0.06))
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if class_idx == 0:
            ax.set_ylabel("GraphLIME attribution score")

    fig.suptitle(
        "Figure 9 - GraphLIME Environmental Attribution by Predicted Affective Class",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.02,
        f"Mean +/- SD across {n_folds} subject-level cross-validation folds; "
        "features follow Equation 9: PM2.5, NDVI, FAR, GVI, NO2, O3.",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    _save(fig, output_dir, filename)
