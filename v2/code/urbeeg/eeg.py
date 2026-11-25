"""EEG preprocessing: one participant in, cleaned stimulus-locked epochs out.

Steps (all per participant, both sessions handled separately then concatenated):
  1. read BrainVision, strip the "BrainVision RDA_" prefix, set ECG/EDA channel types,
     standard 10-20 positions;
  2. restore the online reference (FCz) and re-reference to the common average;
  3. flag and interpolate bad channels (flat or extreme robust variance);
  4. ICA fitted on a 1-100 Hz copy, components labelled with ICLabel, removing
     eye, muscle, heart, line-noise and channel-noise components with p >= 0.8;
  5. band-pass 0.1-40 Hz on the cleaned data, resample to 125 Hz;
  6. epochs -0.2 to 3.0 s around stimulus onset (no baseline correction stored;
     models choose their own), peak-to-peak flag at 150 uV.
"""
from dataclasses import dataclass, asdict
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .config import BIDS_ROOT, EPOCH_TMAX, EPOCH_TMIN, SEED
from .events import trials_from_events

PREFIX = "BrainVision RDA_"
SFREQ_OUT = 125.0
ICLABEL_DROP = {"eye blink", "muscle artifact", "heart beat", "line noise", "channel noise"}
ICLABEL_P = 0.8
PTP_UV = 150.0


@dataclass
class PrepReport:
    participant: str
    session: str
    n_trials: int
    bad_channels: list
    ica_components: int
    ica_removed: dict
    n_ptp_flagged: int


def read_session(vhdr: Path) -> mne.io.Raw:
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="error")
    raw.rename_channels({c: c.replace(PREFIX, "") for c in raw.ch_names})
    # The header writes units as "microvolts", so MNE reads every channel as MISC.
    # Set the types explicitly: ECG, EDA (GSR, unit is really microsiemens), rest EEG.
    types = {}
    for c in raw.ch_names:
        if c.upper() == "ECG":
            types[c] = "ecg"
        elif c.upper().startswith("GSR"):
            types[c] = "misc"
        else:
            types[c] = "eeg"
    raw.set_channel_types(types, verbose="error")
    # MNE does not recognise the unit string "microvolts", so values were read unscaled
    # (in microvolts). Convert EEG and ECG to volts; EDA stays in its own unit (uS).
    raw.apply_function(lambda x: x * 1e-6, picks=["eeg", "ecg"], channel_wise=False)
    mne.add_reference_channels(raw, "FCz", copy=False)   # online reference, restored as zeros
    raw.set_montage(mne.channels.make_standard_montage("colin27_1020"), on_missing="raise",
                    verbose="error")
    raw.set_eeg_reference("average", projection=False, verbose="error")
    return raw


def find_bad_channels(raw: mne.io.Raw, z_thresh: float = 5.0, min_corr: float = 0.4) -> list:
    """Flat channels, channels with extreme robust log-SD (|z| > z_thresh), or channels
    whose highest correlation with any other channel is below min_corr (1-40 Hz).
    The restored reference (FCz) is excluded: it is reconstructed, not recorded."""
    sub = raw.copy().pick("eeg").drop_channels(["FCz"]).filter(1.0, 40.0, verbose="error")
    data, names = sub.get_data(), sub.ch_names
    sd = data.std(axis=1)
    logv = np.log(sd + 1e-20)
    med = np.median(logv)
    mad = np.median(np.abs(logv - med)) * 1.4826 + 1e-12
    z = (logv - med) / mad
    corr = np.corrcoef(data[:, ::5])
    np.fill_diagonal(corr, np.nan)
    max_corr = np.nanmax(np.abs(corr), axis=1)
    return [n for n, s, zz, mc in zip(names, sd, z, max_corr)
            if s < 1e-7 or abs(zz) > z_thresh or mc < min_corr]


def run_ica(raw: mne.io.Raw):
    from mne_icalabel import label_components

    fit_raw = raw.copy().filter(1.0, 100.0, verbose="error")
    ica = mne.preprocessing.ICA(n_components=0.99, method="infomax",
                                fit_params=dict(extended=True), random_state=SEED,
                                max_iter="auto")
    ica.fit(fit_raw, picks="eeg", decim=4, verbose="error")
    labels = label_components(fit_raw, ica, method="iclabel")
    drop, removed = [], {}
    for i, (lab, p) in enumerate(zip(labels["labels"], labels["y_pred_proba"])):
        if lab in ICLABEL_DROP and p >= ICLABEL_P:
            drop.append(i)
            removed[lab] = removed.get(lab, 0) + 1
    ica.exclude = drop
    return ica, removed


def preprocess_session(vhdr: Path, events_tsv: Path):
    sub, ses = vhdr.parts[-4], vhdr.parts[-3]
    trials = trials_from_events(events_tsv)
    raw = read_session(vhdr)

    bads = find_bad_channels(raw)
    raw.info["bads"] = bads
    if bads:
        raw.interpolate_bads(reset_bads=True, verbose="error")

    ica, removed = run_ica(raw)
    ica.apply(raw, verbose="error")

    raw.filter(0.1, 40.0, picks="eeg", verbose="error")
    # Trials whose epoch would run past the end of the recording have a rating but no EEG
    # (sub-91 session a: the recording stops at 29.1 min, the task ran to 33.4 min).
    sf = raw.info["sfreq"]
    onset = trials["onset_sample"].to_numpy()
    has_eeg = (onset + EPOCH_TMIN * sf >= 0) & (onset + EPOCH_TMAX * sf < raw.n_times - 1)
    events = np.column_stack([onset[has_eeg], np.zeros(has_eeg.sum(), int),
                              np.arange(1, has_eeg.sum() + 1)])
    epochs = mne.Epochs(raw, events, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX, baseline=None,
                        picks=["eeg", "ecg", "misc"], preload=True, reject=None,
                        verbose="error")
    if len(epochs) != has_eeg.sum():
        raise RuntimeError(f"{sub} {ses}: {has_eeg.sum()} trials with EEG but {len(epochs)} epochs")
    epochs.resample(SFREQ_OUT, verbose="error")

    eeg = epochs.get_data(picks="eeg")                       # volts
    ptp = np.full(len(trials), np.nan)
    ptp[has_eeg] = (eeg.max(axis=2) - eeg.min(axis=2)).max(axis=1) * 1e6
    trials = trials.assign(participant=sub, session=ses, has_eeg=has_eeg, ptp_uv=ptp,
                           ptp_flag=~has_eeg | (np.nan_to_num(ptp) > PTP_UV))

    report = PrepReport(sub, ses, len(trials), bads, int(ica.n_components_), removed,
                        int((ptp[has_eeg] > PTP_UV).sum()))
    report_extra = {"n_without_eeg": int((~has_eeg).sum())}
    return epochs, trials, report, has_eeg, report_extra


def preprocess_participant(sub: str, out_dir: Path, bids_root: Path = BIDS_ROOT) -> dict:
    """Process both sessions of one participant and save arrays + trial table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    eeg_parts, aux_parts, tables, reports = [], [], [], []
    for vhdr in sorted((bids_root / sub).glob("ses-*/eeg/*_eeg.vhdr")):
        events_tsv = vhdr.with_name(vhdr.name.replace("_eeg.vhdr", "_events.tsv"))
        epochs, trials, report, has_eeg, extra = preprocess_session(vhdr, events_tsv)
        # keep one row per trial: trials without EEG get NaN epochs (never used, see
        # dataset.epochs_per_scene) so that epochs and the trial table stay aligned
        e, a = epochs.get_data(picks="eeg"), epochs.get_data(picks=["ecg", "misc"])
        e_full = np.full((len(trials),) + e.shape[1:], np.nan, np.float32)
        a_full = np.full((len(trials),) + a.shape[1:], np.nan, np.float32)
        e_full[has_eeg], a_full[has_eeg] = e, a
        eeg_parts.append(e_full)
        aux_parts.append(a_full)
        tables.append(trials)
        reports.append({**asdict(report), **extra})
        ch_names = [epochs.ch_names[i] for i in mne.pick_types(epochs.info, eeg=True)]
        times = epochs.times
    np.savez_compressed(out_dir / f"{sub}_epochs.npz",
                        eeg=np.concatenate(eeg_parts), aux=np.concatenate(aux_parts),
                        ch_names=np.array(ch_names), times=times, sfreq=SFREQ_OUT)
    pd.concat(tables, ignore_index=True).to_csv(out_dir / f"{sub}_trials.csv", index=False)
    pd.DataFrame(reports).to_json(out_dir / f"{sub}_report.json", orient="records", indent=1)
    return {"participant": sub, "sessions": len(reports)}
