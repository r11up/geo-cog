# Reproducing the revised manuscript

Every number in the manuscript and the response letter is inserted by `code/scripts/make_results_tex.py`
from the saved results; nothing is typed by hand. This guide maps each table, figure and statistic to the
code that produces it. Run the steps in order. Paths are relative to `v2/`; results are written to
`results/` (set `URBEEG_RESULTS`), derived data to `derived/` (set `URBEEG_DERIVED`).

Data access: EAV is available from its creators on request (data usage agreement) and is not redistributed;
set `EAV_ROOT`. Urban Appraisal is OpenNeuro ds006850 (CC0); set `URBEEG_DATA`. Sentinel-2 imagery is read
through the Microsoft Planetary Computer and OpenStreetMap through Overpass; no account is needed.

## 1. Data preparation

| Step | Command | Output |
|---|---|---|
| EAV grid, features, raw epochs, lockbox list | `python -I code/scripts/prep_eav.py` | `derived/eav/` |
| Urban Appraisal preprocessing (ICA, ICLabel, epochs) | `python -I code/scripts/prep_eeg.py` | `derived/eeg/` |
| Scene images and ratings for the image models | `python -I code/scripts/export_scene_bundle.py` | Kaggle dataset bundle |
| Single-trial bundle for the GPU decoders | `python -I code/scripts/export_kaggle.py` | Kaggle dataset bundle |

## 2. Study 1: EAV (literature-informed prior)

| Paper item | Command | Results folder |
|---|---|---|
| Table 2, Table 3, Table 4 (cross-validation, diagnostic, ablations and controls) | `python -I code/scripts/run_study1_v2.py --protocol cv` | `study1v2_cv_*` |
| Lockbox (12 participants scored once) | `python -I code/scripts/run_study1_v2.py --protocol lockbox` | `study1v2_lockbox_*` |
| Reference pipeline under P1 | `python -I code/scripts/run_study1_v2.py --protocol benchmark` | `study1v2_benchmark_*` |
| Prior quantile and bandwidth sensitivity | `python -I code/scripts/run_contrast.py` | `contrast_*` |
| Equivalence tests (margin +/-3 points) | `python -I code/scripts/equivalence_tests.py` | `equivalence_*` |
| Table 5 (fusion operators, parameter counts) and the Random Forest, LSTM and environment-only rows of Table 2 | `python -I code/scripts/run_study1_fusion.py` after the cross-validation run (same folds, seeds and per-fold hyperparameters; uses a GPU if there is one; also `kaggle/kernel_study1_fusion` with the features in a private dataset) | `study1fusion_cv_*` |
| Holm adjustment over the comparisons of Tables 4 and 5 | `python -I code/scripts/make_results_tex.py` | `study1fusion_cv_*/comparisons_with_reference.csv` |

## 3. Table 6 and Figure 7: published protocols and our methods

| Row | Command | Results folder |
|---|---|---|
| P2, P3, P4 with the reference pipeline | `python -I code/scripts/eav_protocols.py --protocol p2` (also `p3`, `p4`) | `eav_p2_*`, `eav_p3_*`, `eav_p4_*` |
| Further methods under P1, P2, P3, P5 | `python -I code/scripts/eav_newmethods.py --protocol p1` (also `p2`, `p3`, `p5`) | `eav_new_p*_*` |
| EEG foundation model CBraMod under P1 and P3 | `python -I code/scripts/eav_cbramod.py --protocol p1` (also `p3`) | `eav_cbramod_p*_*` |
| Further methods (longer CBraMod schedule, per-participant models, self-training, weighting) | `python -I code/scripts/eav_round5.py --protocol p1` (also `p3`) | `eav_round5_p*_*` |
| Image models, group means (Tan & Deng protocol) | Kaggle kernels `kaggle/kernel_vision`, `kernel_vision3`, `kernel_vision4`, `kernel_vision5` | `vision*/group_benchmark*.csv` |
| Scene-level bootstrap of the group-mean results and paired differences from the reproduced published models | Kaggle kernel `kaggle/kernel_vision_stats` (reads the outputs of the image kernels) | `vision_stats/scene_bootstrap.csv` |
| Figure 7 | `python -I code/scripts/fig_compare.py` (reads `results/table5_forest.csv`, written by `make_results_tex.py`) | `manuscript/figures/fig_compare.pdf` |

The method shown for each published protocol is the candidate with the best validation score across all
rounds; `make_results_tex.py` applies this rule and reports the number and range of candidates. The EAV jobs
can also run on Kaggle (`kaggle/kernel_eav_r5p1`, `kernel_eav_r5p3`, `kernel_eav_p5`) with the
preprocessed epochs in a private dataset.

## 4. Extension: Urban Appraisal

| Paper item | Command | Results folder |
|---|---|---|
| Table 7, Table 8 (linear models, controls) | `python -I code/scripts/run_linear.py` | `linear_*` |
| Table 7, Table 8 (deep models) | `python -I code/scripts/run_deep.py` (GPU; `kaggle/kernel_dualtower`, `kernel_eegnets`) | `deep/` |
| Holm adjustment of the nine primary correlations | `python -I code/scripts/holm_primary.py` | `holm_*` |
| Person-level prediction from image models | `python -I code/scripts/person_level_embeddings.py` (after `kernel_vision`) | `person_embeddings_*` |
| Decoding greenery and built form from EEG | `python -I code/scripts/decode_environment.py` (and `--leakage`) | `decode_*`, `decode_leakage_*` |
| Time-resolved decoding, frozen CBraMod, EEGNet, EEG-Conformer | Kaggle kernel `kaggle/kernel_decode` | `decode_kaggle/` |
| Fine-tuned CBraMod decoding | Kaggle kernel `kaggle/kernel_decode_cbramod` | `decode_kaggle/partD_cbramod_ft.json` |
| Grayscale and phase-scrambled image controls; Mask2Former (ADE20K, Cityscapes) and SegFormer-B5 segmentations | Kaggle kernel `kaggle/kernel_vision6` | `vision6/` |
| Tests of the prior's assumptions, variance partition, residual greenery, design-code check | `python -I code/scripts/prior_validity.py --shares results/vision6/segmentation_shares_v6.csv` | `priorvalidity_*` |

## 5. Figures

| Figure | Command |
|---|---|
| 1, 3, 4 (diagrams) | `python figures/diagrams/make_fig1_overview.py`, `make_fig3_architecture.py`, `make_fig5_protocol.py`, then export with the draw.io CLI |
| 2 (study area) | `python figures/graphics/make_assets.py` |
| 5 (scene composition and its correlation with the ratings) | `python -I code/scripts/fig_scene_composition.py` |
| 6 (decoding over time) | `python -I code/scripts/fig_decoding_time.py` |
| 7 (comparison with published results) | `python -I code/scripts/fig_compare.py` |

## 6. Numbers in the text

`python -I code/scripts/make_results_tex.py` reads the newest finished run of each kind (a run counts as
finished when its folder holds `meta.json` or `summary.csv`) and writes `results_values.tex`, the table rows and
`results/table5_forest.csv`. The analysis plan, with the date of each decision, is in `ANALYSIS-PLAN.md`.

## Seeds and versions

Seeds are fixed in `code/urbeeg/config.py` (`SEED = 20261125`) and derived per fold or repetition from it.
Python 3.11; package versions in `requirements.txt`. Kaggle kernels install `braindecode`, `pyriemann`,
`open_clip_torch` and `timm` at run time.
