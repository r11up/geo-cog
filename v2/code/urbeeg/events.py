"""Turn the BIDS event files into one row per trial.

A trial is: scale prime -> fixation -> stimulus (3 s) -> rating scale -> response.
We key each trial on its stimulus onset and attach the response that follows it.
"""
import re
from pathlib import Path

import pandas as pd

from .config import BIDS_ROOT, REVERSED, SCALE_POINTS, SCALES

_KV = re.compile(r"(\w+):\s*([^;]+)")


def _parse_value(value: str) -> dict:
    """'event: optionConfirmed; scale: SAM-valence; stimulus: X; response: 3' -> dict."""
    return {k.strip(): v.strip() for k, v in _KV.findall(value)}


def recode(scale: str, response: int) -> int:
    """Reverse scales whose left anchor is the 'more' end, so higher = more of the quality."""
    if scale in REVERSED:
        return SCALE_POINTS.get(scale, 5) + 1 - response
    return response


def trials_from_events(events_tsv: Path) -> pd.DataFrame:
    ev = pd.read_csv(events_tsv, sep="\t")
    rows, current = [], None
    for _, e in ev.iterrows():
        kv = _parse_value(str(e["value"]))
        kind = kv.get("event")
        if kind == "anchorsPresented":
            current = {"scale_raw": kv.get("scale")}
        elif kind == "presentedStimulus" and current is not None:
            current.update(stimulus=kv.get("stimulus"), onset_s=float(e["onset"]),
                           onset_sample=int(e["sample"]))
        elif kind == "optionConfirmed" and current is not None and "stimulus" in current:
            # The response marker repeats scale and stimulus; use them as a consistency check.
            if kv.get("stimulus") != current["stimulus"] or kv.get("scale") != current["scale_raw"]:
                raise ValueError(f"{events_tsv.name}: response does not match trial {current}")
            current.update(response=int(kv["response"]), response_s=float(e["onset"]))
            rows.append(current)
            current = None
    df = pd.DataFrame(rows)
    # The beauty marker ("Schönheit") was written with a broken umlaut (e.g. "\x1a");
    # drop anything that is not a letter or hyphen before matching.
    df["scale"] = df["scale_raw"].map(lambda s: SCALES.get(re.sub(r"[^A-Za-z-]", "", s)))
    if df["scale"].isna().any():
        raise ValueError(f"{events_tsv.name}: unknown scale {set(df.loc[df.scale.isna(), 'scale_raw'])}")
    df["rating"] = [recode(s, r) for s, r in zip(df["scale"], df["response"])]
    df["rt_s"] = df["response_s"] - df["onset_s"]
    return df


def load_all_trials(bids_root: Path = BIDS_ROOT) -> pd.DataFrame:
    """All trials of all participants and both sessions (blocks)."""
    frames = []
    for tsv in sorted(bids_root.glob("sub-*/ses-*/eeg/*_events.tsv")):
        sub, ses = tsv.parts[-4], tsv.parts[-3]
        df = trials_from_events(tsv)
        df.insert(0, "session", ses)
        df.insert(0, "participant", sub)
        df["events_file"] = str(tsv.relative_to(bids_root))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["trial_in_session"] = out.groupby(["participant", "session"]).cumcount()
    return out
