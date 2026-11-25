"""Build the EAV inputs used by Study 1: the Astana grid (Sentinel-2 + OpenStreetMap), the EEG
features (DE/PSD, and DE/PSD + tangent-space covariances), preprocessed 5 s epochs for raw-signal
models, and the list of 12 lockbox participants (drawn once with a fixed seed).

    EAV_ROOT=/path/to/EAV ~/.venvs/urbeeg/bin/python -I code/scripts/prep_eav.py
"""
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from urbeeg import eav  # noqa: E402
from urbeeg.astana_grid import build as build_grid  # noqa: E402
from urbeeg.config import DERIVED  # noqa: E402


def main():
    out = DERIVED / "eav"
    out.mkdir(parents=True, exist_ok=True)
    steps = [("astana_grid.csv", lambda p: build_grid(p)), ("eav_features.npz", eav.build),
             ("eav_features_v2.npz", eav.build_v2), ("eav_raw.npy", eav.build_raw)]
    for name, fn in steps:
        if not (out / name).exists():
            print("building", name, flush=True)
            fn(out / name)
    lock = out / "lockbox_subjects.txt"
    if not lock.exists():
        pick = np.random.default_rng(20261125).choice(np.arange(1, 43), 12, replace=False)
        lock.write_text(" ".join(map(str, sorted(pick.tolist()))) + "\n")
    print("lockbox:", lock.read_text().strip())


if __name__ == "__main__":
    main()
