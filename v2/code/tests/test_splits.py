import itertools
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urbeeg.splits import crossed_cells, inner_validation  # noqa: E402

PARTS = [f"sub-{i:02d}" for i in range(63)]
SCENES = [f"{a}{b}{i:02d}" for a in ("HT", "LT") for b in ("HB", "LB") for i in range(14)]
STRATA = {s: s[:4] for s in SCENES}


def cells():
    return crossed_cells(PARTS, SCENES, STRATA, n_p=5, n_s=4, n_rep=3, seed=7)


def test_no_person_or_scene_in_both_train_and_test():
    for c in cells():
        assert not set(c.train_participants) & set(c.test_participants)
        assert not set(c.train_scenes) & set(c.test_scenes)
        assert set(c.train_participants) | set(c.test_participants) == set(PARTS)
        assert set(c.train_scenes) | set(c.test_scenes) == set(SCENES)


def test_every_pair_tested_exactly_once_per_repetition():
    for rep in range(3):
        seen = Counter()
        for c in cells():
            if c.rep == rep:
                seen.update(itertools.product(c.test_participants, c.test_scenes))
        assert len(seen) == len(PARTS) * len(SCENES)
        assert set(seen.values()) == {1}


def test_scene_folds_are_balanced_across_design_codes():
    for c in cells():
        counts = Counter(STRATA[s] for s in c.test_scenes)
        assert set(counts) == {"HTHB", "HTLB", "LTHB", "LTLB"}
        assert max(counts.values()) - min(counts.values()) <= 1


def test_repetitions_differ_and_are_reproducible():
    a, b = cells(), cells()
    assert a == b
    reps = {c.rep: c.test_participants for c in a if c.p_fold == 0 and c.s_fold == 0}
    assert len(set(reps.values())) == 3


def test_inner_validation_is_disjoint_and_inside_training():
    for c in cells():
        tr_p, va_p, tr_s, va_s = inner_validation(c)
        assert not set(tr_p) & set(va_p) and not set(tr_s) & set(va_s)
        assert set(tr_p) | set(va_p) == set(c.train_participants)
        assert set(tr_s) | set(va_s) == set(c.train_scenes)
