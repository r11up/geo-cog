"""Which participants are fully downloaded? Compares local file sizes with the
OpenNeuro S3 manifest (derived/s3_manifest.tsv: relative path <TAB> size in bytes)."""
from pathlib import Path

import pandas as pd

from .config import BIDS_ROOT, DERIVED


def manifest(path: Path = DERIVED / "s3_manifest.tsv") -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", names=["path", "size"])


def complete_participants(bids_root: Path = BIDS_ROOT) -> tuple[list, list]:
    m = manifest()
    m = m[m["path"].str.startswith("sub-")]
    m["participant"] = m["path"].str.split("/").str[0]
    done, pending = [], []
    for sub, files in m.groupby("participant"):
        ok = all((bids_root / p).is_file() and (bids_root / p).stat().st_size == s
                 for p, s in zip(files["path"], files["size"]))
        (done if ok else pending).append(sub)
    return sorted(done), sorted(pending)
