"""Classification metrics, fold summaries, significance tests, and result printing."""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
)
from scipy.stats import wilcoxon


@dataclass
class MetricResult:
    accuracy: float
    macro_f1: float
    roc_auc: float
    confusion_mat: np.ndarray


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int = 5,
) -> MetricResult:
    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob,
                                   multi_class="ovr", average="macro"))
    except ValueError:
        auc = 0.5
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    return MetricResult(accuracy=acc, macro_f1=f1, roc_auc=auc,
                        confusion_mat=cm)


def aggregate_cv_metrics(
    fold_results: List[MetricResult],
) -> Dict[str, Dict[str, float]]:
    out: Dict = {}
    for metric in ("accuracy", "macro_f1", "roc_auc"):
        vals = np.array([getattr(r, metric) for r in fold_results])
        out[metric] = {
            "mean": float(vals.mean()),
            "std":  float(vals.std()),
            "folds": vals.tolist(),
        }
    return out


def wilcoxon_test(scores_a: List[float], scores_b: List[float]) -> float:
    stat, p_val = wilcoxon(scores_a, scores_b, method='exact')
    return float(p_val)


def print_results_table(results_dict: Dict[str, Dict], class_names=None):
    header = f"{'Model':<45} {'Accuracy (%)':>14} {'Macro F1':>10} {'ROC-AUC':>10}"
    print(header)
    print("─" * len(header))
    for model_name, metrics in results_dict.items():
        acc_m  = metrics["accuracy"]["mean"] * 100
        acc_s  = metrics["accuracy"]["std"]  * 100
        f1_m   = metrics["macro_f1"]["mean"]
        f1_s   = metrics["macro_f1"]["std"]
        auc_m  = metrics["roc_auc"]["mean"]
        auc_s  = metrics["roc_auc"]["std"]
        print(
            f"{model_name:<45} "
            f"{acc_m:>7.1f} ± {acc_s:.1f}  "
            f"{f1_m:.3f} ± {f1_s:.3f}  "
            f"{auc_m:.3f} ± {auc_s:.3f}"
        )
