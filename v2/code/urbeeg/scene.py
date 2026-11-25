"""Scene composition features for the 56 Street View stimuli.

Source: the dataset authors' SegFormer-B5 (ADE20K) pixel counts per class
(analysis-repo/data/segmentation.zip, segmentation_pixel_sums_incl.csv), which
covers all 56 images. Classes are grouped into the categories used in urban
streetscape research; each feature is that category's share of the image.
"""
import io
import zipfile
from pathlib import Path

import pandas as pd

from .config import DATA_ROOT

SEG_ZIP = DATA_ROOT / "analysis-repo/data/segmentation.zip"
SEG_CSV = "segmentation_pixel_sums_incl.csv"

GROUPS = {
    "greenery": ["tree", "grass", "plant", "flower", "field", "palm"],
    "buildings": ["building", "wall", "house", "skyscraper", "tower", "column", "door",
                  "awning", "stairs", "step", "ceiling"],
    "sky": ["sky"],
    "ground": ["road", "sidewalk", "earth", "base", "path"],
    "vehicles": ["car", "van", "truck", "bus", "minibike", "bicycle"],
    "people": ["person"],
}
FEATURES = list(GROUPS) + ["other"]


def load_pixel_counts(seg_zip: Path = SEG_ZIP) -> pd.DataFrame:
    with zipfile.ZipFile(seg_zip) as z:
        raw = z.read(SEG_CSV).decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), sep=";")
    return df.rename(columns={"StimulusId": "stimulus"}).drop(columns=["Stimulus"]).set_index("stimulus")


def composition(counts: pd.DataFrame) -> pd.DataFrame:
    """Share of each grouped category; 'other' collects every remaining ADE20K class."""
    counts = counts.rename(columns=lambda c: c.replace(".", "_"))
    total = counts.sum(axis=1)
    out = pd.DataFrame(index=counts.index)
    used = set()
    for name, classes in GROUPS.items():
        cols = [c for c in classes if c in counts.columns]
        used.update(cols)
        out[name] = counts[cols].sum(axis=1) / total
    out["other"] = counts[[c for c in counts.columns if c not in used]].sum(axis=1) / total
    out["green_view_index"] = out["greenery"]
    out["sky_view"] = out["sky"]
    return out.sort_index()


LLVF_ZIP = DATA_ROOT / "analysis-repo/data/llvfs.zip"
LOW_LEVEL = ["Brightness", "sdBrightness", "Saturation", "sdSaturation", "HSVHue",
             "StraightEdgeDensity", "NonStraightEdgeDensity", "Entropy", "FractalD",
             "depth_metric_mean", "depth_metric_sd"]


def low_level_features(llvf_zip: Path = LLVF_ZIP) -> pd.DataFrame:
    """The dataset authors' low-level image statistics and monocular depth estimates.
    Used only as a control: does composition predict appraisal beyond these?"""
    with zipfile.ZipFile(llvf_zip) as z:
        llvf = pd.read_csv(io.BytesIO(z.read("llvf_file.csv"))).set_index("ImageName")
        depth = pd.read_csv(io.BytesIO(z.read("depth_file.csv"))).set_index("file_name")
    out = llvf.join(depth, how="inner")[LOW_LEVEL]
    out.index.name = "stimulus"
    return out.sort_index()


def scene_features() -> pd.DataFrame:
    feats = composition(load_pixel_counts())
    # Stimulus-name design codes. HT vs LT matches greenery (HT mean share 0.18-0.27,
    # LT ~0.00). HB vs LB does NOT differ in building share, so its meaning is not
    # assumed here; it is kept only as a stratification label for the scene folds.
    feats["code_HT"] = feats.index.str.startswith("HT").astype(int)
    feats["code_HB"] = (feats.index.str[2:4] == "HB").astype(int)
    return feats
