"""Code-correctness tests on SYNTHETIC data with a known built-in relationship.
These numbers check the machinery only; they are never reported."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urbeeg.evaluate import cell_targets, run_cells  # noqa: E402
from urbeeg.metrics import compare, per_person_r, summarize  # noqa: E402
from urbeeg.models_linear import model_zoo  # noqa: E402
from urbeeg.scene import FEATURES, LOW_LEVEL  # noqa: E402
from urbeeg.splits import crossed_cells  # noqa: E402

TARGETS = ["valence", "arousal", "stress"]


def synthetic(n_p=24, n_s=32, n_ch=6, seed=0):
    rng = np.random.default_rng(seed)
    parts = [f"sub-{i:02d}" for i in range(n_p)]
    scenes = [f"{a}{b}{i:02d}" for a in ("HT", "LT") for b in ("HB", "LB") for i in range(n_s // 4)]
    scene = pd.DataFrame(rng.random((len(scenes), len(FEATURES))), index=scenes, columns=FEATURES)
    scene[FEATURES] = scene[FEATURES].div(scene[FEATURES].sum(1), axis=0)
    for c in LOW_LEVEL:
        scene[c] = rng.normal(size=len(scenes))
    P, S = np.meshgrid(parts, scenes, indexing="ij")
    P, S = P.ravel(), S.ravel()
    green = scene.loc[S, "greenery"].to_numpy()
    person_off = dict(zip(parts, rng.normal(0, 1, n_p)))
    idio = rng.normal(0, 1, len(P))                         # person-specific deviation
    valence = 5 + 8 * (green - green.mean()) + np.array([person_off[p] for p in P]) + idio
    ratings = pd.DataFrame({"valence": valence,
                            "arousal": rng.normal(5, 1, len(P)),
                            # stress noise 0.1 vs scene signal SD ~0.29: best possible r ~0.95
                            "stress": 3 - 4 * (green - green.mean()) + rng.normal(0, .1, len(P))})
    erp_comp = rng.normal(size=(len(P), 4))
    erp_comp[:, 0] += 2 * idio                              # EEG carries the deviation
    erp_win = rng.normal(size=(len(P), 16))
    erp_win[:, 0] += 2 * idio                               # ... in both EEG feature sets
    A = rng.normal(size=(len(P), 3, n_ch, n_ch))
    cov = np.einsum("nbij,nbkj->nbik", A, A) + 3 * np.eye(n_ch)
    return {"participant": P, "stimulus": S, "erp_comp": erp_comp,
            "erp_win": erp_win, "cov": cov,
            "ratings": ratings, "scene": scene}, parts, scenes


@pytest.fixture(scope="module")
def results():
    data, parts, scenes = synthetic()
    cells = crossed_cells(parts, scenes, {s: s[:4] for s in scenes}, n_p=4, n_s=4, n_rep=1, seed=3)
    raw = run_cells(data, cells, model_zoo, TARGETS, mode="raw", log=lambda *_: None)
    dev = run_cells(data, cells, model_zoo, TARGETS, mode="deviation", log=lambda *_: None)
    return data, cells, pd.concat([raw, dev])


def _r(summary, model, target, mode):
    row = summary[(summary.model == model) & (summary.target == target) & (summary["mode"] == mode)]
    return float(row["r"].iloc[0])


def test_scene_model_recovers_scene_effect_and_controls_do_not(results):
    _, _, oof = results
    s = summarize(oof, n_boot=200)
    assert _r(s, "scene", "stress", "raw") > 0.8
    assert abs(_r(s, "ctrl_scene_noise", "stress", "raw")) < 0.35
    assert abs(_r(s, "ctrl_scene_shuffled", "stress", "raw")) < 0.35
    assert _r(s, "intercept", "stress", "raw") == 0.0


def test_eeg_predicts_personal_deviation_and_shuffled_eeg_does_not(results):
    _, _, oof = results
    s = summarize(oof, n_boot=200)
    assert _r(s, "eeg_erp4", "valence", "deviation") > 0.5
    assert abs(_r(s, "ctrl_eeg_shuffled", "valence", "deviation")) < 0.35
    assert abs(_r(s, "scene", "valence", "deviation")) < 0.35


def test_fusion_beats_each_part_on_raw_valence(results):
    _, _, oof = results
    s = summarize(oof, n_boot=200)
    f = _r(s, "fusion", "valence", "raw")
    assert f > _r(s, "scene", "valence", "raw") and f > _r(s, "eeg_erp4", "valence", "raw") - 0.05
    c = compare(oof, "fusion", "scene")
    assert c.query("target == 'valence' and mode == 'raw'")["mean_dz"].iloc[0] > 0


def test_every_pair_predicted_once_per_rep_per_model(results):
    data, _, oof = results
    n_pairs = len(data["participant"])
    counts = oof.groupby(["model", "target", "mode", "rep"]).size()
    assert set(counts) == {n_pairs}


def test_test_ratings_never_define_their_own_targets(results):
    data, cells, _ = results
    cell = cells[0]
    base = cell_targets(data, cell, TARGETS, "deviation")
    # change one test person's rating of one test scene
    i = np.where(np.isin(data["participant"], cell.test_participants)
                 & np.isin(data["stimulus"], cell.test_scenes))[0][0]
    d2 = dict(data, ratings=data["ratings"].copy())
    d2["ratings"].iloc[i, 0] += 50
    changed = cell_targets(d2, cell, TARGETS, "deviation")
    diff = np.where(np.abs(changed[:, 0] - base[:, 0]) > 1e-9)[0]
    assert list(diff) == [i]          # only that row's own target moves


def test_per_person_r_is_one_for_perfect_predictions():
    oof = pd.DataFrame({"model": "m", "target": "t", "mode": "raw", "rep": 0,
                        "participant": np.repeat(["a", "b"], 10), "stimulus": list(range(10)) * 2,
                        "y_true": np.r_[np.arange(10), np.arange(10)[::-1]]})
    oof["y_pred"] = oof["y_true"] * 3 + 7
    assert np.allclose(per_person_r(oof)["r"], 1.0, atol=1e-3)


def test_missing_ratings_are_imputed_for_training_and_skipped_for_scoring():
    data, parts, scenes = synthetic(n_p=12, n_s=16)
    data["ratings"].iloc[3, 0] = np.nan
    data["ratings"].iloc[40, 2] = np.nan
    cells = crossed_cells(parts, scenes, {s: s[:4] for s in scenes}, n_p=4, n_s=4, n_rep=1, seed=1)
    for mode in ("raw", "deviation"):
        oof = run_cells(data, cells, model_zoo, TARGETS, mode=mode, log=lambda *_: None)
        assert not oof["y_pred"].isna().any()
        assert summarize(oof, n_boot=50)["r"].notna().all()
