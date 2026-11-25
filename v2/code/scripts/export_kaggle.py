"""Compact bundle for running the deep models on Kaggle: float16 microvolt epochs, trial
tables, pair features, the scene table and the code. Writes derived/kaggle_bundle/."""
import shutil
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import numpy as np  # noqa: E402

from urbeeg.config import DERIVED  # noqa: E402
from urbeeg.dataset import FEAT_DIR, EEG_DIR, cache_features, processed_participants  # noqa: E402
from urbeeg.scene import low_level_features, scene_features  # noqa: E402

out = DERIVED / "kaggle_bundle"
(out / "eeg").mkdir(parents=True, exist_ok=True)
(out / "features").mkdir(exist_ok=True)
subs = processed_participants()
cache_features(subs)
for s in subs:
    z = np.load(EEG_DIR / f"{s}_epochs.npz")
    np.savez_compressed(out / "eeg" / f"{s}_epochs.npz", eeg_uv=(z["eeg"] * 1e6).astype(np.float16),
                        ch_names=z["ch_names"], times=z["times"], sfreq=z["sfreq"])
    shutil.copy(EEG_DIR / f"{s}_trials.csv", out / "eeg")
    shutil.copy(EEG_DIR / f"{s}_report.json", out / "eeg")
    shutil.copy(FEAT_DIR / f"{s}_pairfeat.npz", out / "features")
scene_features().join(low_level_features(), how="left").to_csv(out / "scene_features.csv")
shutil.copytree(CODE / "urbeeg", out / "code" / "urbeeg", dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(CODE / "scripts", out / "code" / "scripts", dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__"))
print(len(subs), "participants exported to", out)
