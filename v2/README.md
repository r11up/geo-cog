# geo-cog v2: revision code

Code for the revised manuscript *Neuro-Geospatial Modelling of EEG Affective States: A Test of
Literature-Informed Environmental Priors, Protocol-Matched Benchmarks and a Co-Registered Extension*. Version 1 (the repository root) is the
code of the original submission and is kept unchanged.

## What v2 adds

**Study 1, EAV (laboratory EEG, no co-registration).** A re-implementation of the dual-tower model:
an EEG tower over channel tokens of differential-entropy and power-spectral-density features plus a
token of band-wise spatial covariances in Riemannian tangent space (re-centred per participant), a
graph tower over a 10 x 10 km grid of Astana in 1 km cells, and bidirectional cross-modal attention.
Environmental layers come from Sentinel-2 L2A (Microsoft Planetary Computer) and OpenStreetMap
(Overpass). The environmental vector of each epoch is drawn either without the label (label-free
pairing) or from the literature-informed prior (PECM), with input controls: pure noise, matched random
vectors with the prior's class moments, environment removed at test, random pairing, shuffled labels,
reversed prior, and a label-conditioned diagnostic. Grid scales of 1, 2 and 5 km. Evaluation:
subject-level cross-validation, a lockbox of 12 participants held out from all model selection, and
the published within-participant chronological protocol of the EAV paper. Every run is appended to
`results/tracker.tsv`.

**Study 2, Urban Appraisal (OpenNeuro ds006850, co-registered).** EEG preprocessing (ICA with
ICLabel), scene composition from semantic segmentation, low-level image statistics, linear models
(ERP, Riemannian, stacking), deep models (EEGNet, EEG-Conformer, a dual tower with token-level
cross-attention) and controls, evaluated on unseen people x unseen scenes with crossed folds.

## Layout

```
v2/
  code/urbeeg/      package: EEG and scene features, grids, models, splits, metrics, tracker
  code/scripts/     entry points (preprocessing, experiments, LaTeX value export)
  code/tests/       pytest suite (synthetic data)
  figures/          draw.io diagrams generated from Python, data graphics
  kaggle/           GPU runners: Study 2 deep models, vision benchmark, decoding
  requirements.txt
```

## Data

- **EAV**: request access through the dataset's Zenodo record (data usage agreement). The data are
  not redistributed here. Set `EAV_ROOT` to the local copy.
- **Urban Appraisal**: OpenNeuro ds006850 and the authors' stimulus repository. Set `URBEEG_DATA`.
- **Sentinel-2 / OpenStreetMap**: read on the fly (no account needed).

Derived files and results are written to `v2/derived/` and `v2/results/` (override with
`URBEEG_DERIVED` and `URBEEG_RESULTS`).

## Running

Python 3.11; `pip install -r requirements.txt`.

```bash
# Study 1
python -I code/scripts/prep_eav.py                       # grid, features, raw epochs, lockbox list
python -I code/scripts/eav_explore.py                    # EEG pipelines, development participants
python -I code/scripts/eav_benchmark.py --variants logreg_both_pooled hgb_both_pooled
python -I code/scripts/run_study1_v2.py --protocol cv    # also: lockbox, benchmark
python -I code/scripts/eav_protocols.py --protocol p2    # published protocols: p2, p3, p4
python -I code/scripts/eav_newmethods.py --protocol p1   # further methods on the published splits (p1, p2, p3);
                                                         # the reported one is chosen on validation data
python -I code/scripts/eav_cbramod.py --protocol p1      # EEG foundation model CBraMod (p1 fine-tuned, p3 frozen)

# Study 2
python -I code/scripts/prep_eeg.py
python -I code/scripts/run_linear.py
python -I code/scripts/run_deep.py                       # GPU; see kaggle/
python -I code/scripts/decode_environment.py             # decode scene composition from EEG
python -I code/scripts/export_scene_bundle.py            # images + ratings for kaggle/kernel_vision
python -I code/scripts/person_level_embeddings.py        # after kernel_vision: unseen people x scenes
# kaggle/kernel_vision3: larger encoders, mirrored-image averaging and language-anchored features
# for the group-level appraisal benchmark (pre-specified equal-weight ensemble)
python -I code/scripts/fig_decoding_time.py              # after kaggle/kernel_decode

# Figures and tests
python figures/graphics/make_assets.py
python figures/diagrams/make_fig3_architecture.py        # then export with the draw.io CLI
cd code && python -I -m pytest -q tests
```

## Licence

As for the repository (see `LICENSE`).
