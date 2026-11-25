"""Paths and fixed study constants for the Urban Appraisal analysis.

Everything that defines the study design lives here, in one place.
"""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("URBEEG_DATA", Path.home() / "Desktop/Research/datasets/UrbanAppraisal-EEG"))
BIDS_ROOT = DATA_ROOT / "openneuro-ds006850"
STIMULI_DIRS = [
    DATA_ROOT / "stimuli-repo/block1/ressources/stimuli",
    DATA_ROOT / "stimuli-repo/block2/ressources/stimuli",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]          # geo-cog/v2
DERIVED = Path(os.environ.get("URBEEG_DERIVED", PROJECT_ROOT / "derived"))  # cached data
RESULTS = Path(os.environ.get("URBEEG_RESULTS", PROJECT_ROOT / "results"))

# Rating scales as written in the event markers -> short English name.
SCALES = {
    "SAM-valence": "valence",
    "SAM-arousal": "arousal",
    "SAM-dominance": "dominance",
    "Likert-Stress": "stress",
    "Likert-Offenheit": "openness",
    "Likert-Sicherheit": "safety",
    "Likert-Schnheit": "beauty",        # "Schönheit" lost its umlaut in the markers
    "Likert-Heimeligkeit": "hominess",
    "Likert-Faszination": "fascination",
    # sub-51 session a was run with an English-labelled version of the experiment script.
    "Likert-stress": "stress",
    "Likert-openness": "openness",
    "Likert-safety": "safety",
    "Likert-beauty": "beauty",
    "Likert-hominess": "hominess",
    "Likert-fascination": "fascination",
}

# Scale length and whether the raw response must be reversed so that a higher
# value means more of the named quality. Anchors (dataset README, left - right):
# valence happy-unhappy, arousal excited-calm, dominance controlled-in control,
# stress relaxed-stressful, openness narrow-open, safety unsafe-safe,
# beauty ugly-beautiful, hominess alienated-at home, fascination boring-fascinating.
SCALE_POINTS = {"valence": 9, "arousal": 9, "dominance": 9}          # others are 5-point
REVERSED = {"valence", "arousal"}

PRIMARY_TARGETS = ["valence", "arousal", "stress"]
SECONDARY_TARGETS = ["dominance", "openness", "safety", "beauty", "hominess", "fascination"]

# Electrode clusters used by the dataset authors for their ERP components.
OCCIPITAL = ["POz", "Oz", "O1", "O2"]          # P1, N1
PARIETAL = ["CPz", "Pz", "P3", "P4"]           # P3, LPP

EPOCH_TMIN, EPOCH_TMAX = -0.2, 3.0             # seconds around stimulus onset

# Participants excluded from all analyses, with the reason (reported in Methods).
EXCLUDED = {
    "sub-91": "EEG file of block 2 on OpenNeuro ends after 8.5 of 29.1 min and block 1 "
              "after 29.1 of 33.4 min; 222 of 504 rated trials have no EEG",
}
SEED = 20261125
