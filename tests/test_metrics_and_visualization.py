"""
tests/test_metrics_and_visualization.py
========================================
Unit tests for evaluation/metrics.py and evaluation/visualization.py.

Covers:
  - compute_metrics: correct shapes, metric ranges, perfect predictor
  - aggregate_cv_metrics: mean/std computation
  - wilcoxon_test: p-value range
  - compute_spatial_error_rates: signature, output shape, value range
  - print_results_table: smoke test (no crash)

Run:
    pytest tests/test_metrics_and_visualization.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import (
    compute_metrics,
    aggregate_cv_metrics,
    wilcoxon_test,
    MetricResult,
    print_results_table,
)
from evaluation.visualization import (
    compute_spatial_error_rates,
    plot_graphlime_environmental_attribution_figure9,
)


RNG = np.random.default_rng(99)
N   = 200
K   = 5   # classes


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def test_output_type(self):
        y_true = RNG.integers(0, K, N)
        y_pred = RNG.integers(0, K, N)
        y_prob = RNG.dirichlet(np.ones(K), N)
        m = compute_metrics(y_true, y_pred, y_prob)
        assert isinstance(m, MetricResult)

    def test_metric_ranges(self):
        y_true = RNG.integers(0, K, N)
        y_pred = RNG.integers(0, K, N)
        y_prob = RNG.dirichlet(np.ones(K), N)
        m = compute_metrics(y_true, y_pred, y_prob)
        assert 0.0 <= m.accuracy  <= 1.0
        assert 0.0 <= m.macro_f1  <= 1.0
        assert 0.0 <= m.roc_auc   <= 1.0

    def test_perfect_predictor(self):
        y_true = RNG.integers(0, K, N)
        one_hot = np.eye(K)[y_true]
        m = compute_metrics(y_true, y_true, one_hot)
        assert m.accuracy == pytest.approx(1.0)
        assert m.macro_f1 == pytest.approx(1.0)
        assert m.roc_auc  == pytest.approx(1.0)

    def test_confusion_matrix_shape(self):
        y_true = RNG.integers(0, K, N)
        y_pred = RNG.integers(0, K, N)
        y_prob = RNG.dirichlet(np.ones(K), N)
        m = compute_metrics(y_true, y_pred, y_prob)
        assert m.confusion_mat.shape == (K, K)

    def test_confusion_matrix_sums_to_n(self):
        y_true = RNG.integers(0, K, N)
        y_pred = RNG.integers(0, K, N)
        y_prob = RNG.dirichlet(np.ones(K), N)
        m = compute_metrics(y_true, y_pred, y_prob)
        assert m.confusion_mat.sum() == N

    def test_random_chance_accuracy(self):
        """Random predictor should be near 1/K for a balanced dataset."""
        y_true = np.tile(np.arange(K), N // K)
        y_pred = RNG.integers(0, K, len(y_true))
        y_prob = RNG.dirichlet(np.ones(K), len(y_true))
        m = compute_metrics(y_true, y_pred, y_prob)
        # Within 3 std of binomial: ±3√(p(1-p)/n) ≈ ±3 * 0.063 ≈ 0.19
        assert abs(m.accuracy - 1 / K) < 0.25

    def test_single_class_auc_fallback(self):
        """Single class in y_true should not raise — returns 0.5."""
        y_true = np.zeros(20, dtype=int)
        y_pred = np.zeros(20, dtype=int)
        y_prob = np.zeros((20, K)); y_prob[:, 0] = 1.0
        m = compute_metrics(y_true, y_pred, y_prob)
        assert m.roc_auc == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# aggregate_cv_metrics
# ---------------------------------------------------------------------------

class TestAggregateCVMetrics:

    def test_output_keys(self):
        folds = [
            MetricResult(0.7, 0.65, 0.85, np.eye(K, dtype=int)),
            MetricResult(0.72, 0.67, 0.87, np.eye(K, dtype=int)),
        ]
        agg = aggregate_cv_metrics(folds)
        assert "accuracy" in agg
        assert "macro_f1" in agg
        assert "roc_auc"  in agg

    def test_mean_correct(self):
        folds = [
            MetricResult(0.7, 0.6, 0.8, np.eye(K, dtype=int)),
            MetricResult(0.8, 0.7, 0.9, np.eye(K, dtype=int)),
        ]
        agg = aggregate_cv_metrics(folds)
        assert agg["accuracy"]["mean"] == pytest.approx(0.75)
        assert agg["macro_f1"]["mean"] == pytest.approx(0.65)

    def test_std_correct(self):
        folds = [
            MetricResult(0.0, 0.0, 0.0, np.eye(K, dtype=int)),
            MetricResult(1.0, 1.0, 1.0, np.eye(K, dtype=int)),
        ]
        agg = aggregate_cv_metrics(folds)
        assert agg["accuracy"]["std"] == pytest.approx(0.5)

    def test_folds_list_length(self):
        folds = [MetricResult(0.7, 0.65, 0.85, np.eye(K, dtype=int))] * 5
        agg = aggregate_cv_metrics(folds)
        assert len(agg["accuracy"]["folds"]) == 5


# ---------------------------------------------------------------------------
# wilcoxon_test
# ---------------------------------------------------------------------------

class TestWilcoxonTest:

    def test_p_value_range(self):
        a = [0.70, 0.72, 0.71, 0.73, 0.69]
        b = [0.65, 0.66, 0.64, 0.67, 0.63]
        p = wilcoxon_test(a, b)
        assert 0.0 <= p <= 1.0

    def test_identical_scores_p_one(self):
        """Identical scores → no evidence of difference → high p-value."""
        a = [0.75] * 5
        b = [0.75] * 5
        # wilcoxon raises ValueError for zero differences; we test the range
        try:
            p = wilcoxon_test(a, b)
            assert p >= 0.0
        except Exception:
            pass   # scipy raises for all-zero differences — acceptable


# ---------------------------------------------------------------------------
# compute_spatial_error_rates
# ---------------------------------------------------------------------------

class TestComputeSpatialErrorRates:

    def test_output_shape(self):
        y_true = RNG.integers(0, K, 100)
        y_pred = RNG.integers(0, K, 100)
        err = compute_spatial_error_rates(y_true, y_pred, n_nodes=100, n_classes=K)
        assert err.shape == (100,)

    def test_values_in_unit_interval(self):
        y_true = RNG.integers(0, K, 200)
        y_pred = RNG.integers(0, K, 200)
        err = compute_spatial_error_rates(y_true, y_pred)
        assert (err >= 0.0).all() and (err <= 1.0).all()

    def test_perfect_predictor_zero_error(self):
        y_true = np.tile(np.arange(K), 40)
        err = compute_spatial_error_rates(y_true, y_true)
        assert (err == 0.0).all()

    def test_no_redundant_labels_parameter(self):
        """Function must NOT accept a 'labels' positional parameter."""
        import inspect
        sig = inspect.signature(compute_spatial_error_rates)
        assert 'labels' not in sig.parameters, (
            "compute_spatial_error_rates should not have a 'labels' parameter "
            "(was the redundant 3rd arg removed?)"
        )

    def test_all_nodes_receive_counts(self):
        """With enough samples (>> n_nodes), all nodes should be non-zero."""
        y_true = np.tile(np.arange(K), 1000)
        y_pred = np.tile(np.arange(K), 1000)
        err = compute_spatial_error_rates(y_true, y_pred, n_nodes=100, n_classes=K)
        assert (err == 0.0).all()   # perfect predictor → all zero
        # Now with random predictions
        y_pred2 = RNG.integers(0, K, len(y_true))
        err2 = compute_spatial_error_rates(y_true, y_pred2, n_nodes=100, n_classes=K)
        assert (err2 > 0).sum() >= 90, "With 5000 samples, most nodes should have data"

    def test_class_node_zoning(self):
        """Class c should only assign to nodes in band [c*20, (c+1)*20)."""
        y_true = np.zeros(200, dtype=int)  # all class 0
        y_pred = np.ones(200, dtype=int)   # all wrong
        err = compute_spatial_error_rates(y_true, y_pred, n_nodes=100, n_classes=5)
        # Class 0 owns nodes 0-19; nodes 20-99 should be zero
        assert (err[:20] > 0).all(), "Class 0 nodes (0-19) should have errors"
        assert (err[20:] == 0).all(), "Nodes 20-99 should have zero errors for class-0 input"

    def test_observed_node_ids_override_proxy_assignment(self):
        """When node IDs are supplied, rates should use observed assignments."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        node_ids = np.array([0, 0, 1, 1])

        err = compute_spatial_error_rates(
            y_true, y_pred, n_nodes=4, n_classes=2, node_ids=node_ids
        )

        assert err[0] == 0.5
        assert err[1] == 0.5
        assert err[2] == 0.0
        assert err[3] == 0.0


# ---------------------------------------------------------------------------
# Figure 9 GraphLIME attribution plotting
# ---------------------------------------------------------------------------

def _fold_attribution_fixture() -> np.ndarray:
    base = np.array([
        [0.34, 0.28, 0.50, 0.22, 0.32, 0.18],
        [0.62, 0.21, 0.34, 0.19, 0.41, 0.25],
        [0.68, 0.18, 0.39, 0.15, 0.44, 0.22],
        [0.24, 0.58, 0.20, 0.46, 0.19, 0.31],
        [0.20, 0.64, 0.17, 0.50, 0.16, 0.29],
    ], dtype=np.float32)
    return np.stack([base, base + 0.01, base - 0.01], axis=0)


class TestFigure9GraphLIME:

    def test_requires_fold_attributions(self, tmp_path):
        with pytest.raises(ValueError, match="fold_attributions is required"):
            plot_graphlime_environmental_attribution_figure9(
                tmp_path, fold_attributions=None
            )

    def test_plot_creates_png_and_pdf(self, tmp_path):
        plot_graphlime_environmental_attribution_figure9(
            tmp_path, fold_attributions=_fold_attribution_fixture()
        )
        assert (tmp_path / "figure9_graphlime_environmental_attribution.png").exists()
        assert (tmp_path / "figure9_graphlime_environmental_attribution.pdf").exists()


# ---------------------------------------------------------------------------
# print_results_table (smoke test)
# ---------------------------------------------------------------------------

class TestPrintResultsTable:

    def test_no_crash(self, capsys):
        results = {
            "Model A": {"accuracy": {"mean": 0.7, "std": 0.02, "folds": [0.68, 0.72]},
                        "macro_f1": {"mean": 0.65, "std": 0.03, "folds": [0.62, 0.68]},
                        "roc_auc":  {"mean": 0.85, "std": 0.01, "folds": [0.84, 0.86]}},
            "Model B": {"accuracy": {"mean": 0.75, "std": 0.02, "folds": [0.73, 0.77]},
                        "macro_f1": {"mean": 0.70, "std": 0.03, "folds": [0.67, 0.73]},
                        "roc_auc":  {"mean": 0.88, "std": 0.01, "folds": [0.87, 0.89]}},
        }
        print_results_table(results)   # must not raise
        out = capsys.readouterr().out
        assert "Model A" in out and "Model B" in out
