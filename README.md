# Neuro-Geospatial Affective Computing 

> **Version 2 (2026 revision).** The code for the revised manuscript is in [`v2/`](v2/README.md). Everything below describes version 1, the code of the original submission, which is kept unchanged.


This repository contains the code for the paper "A Neuro-Geospatial Framework for Integrating Environmental Context into EEG-Based Affective Computing".

---

## Overview

This repository implements an end-to-end neuro-geospatial pipeline that combines EEG features with live city-level environmental context indicators to predict affective mental states.

| Step | Description | Key file |
|------|-------------|----------|
| 1 | EEG preprocessing (bandpass, ASR, ICA, epoch seg.) | `data/eeg_preprocessing.py` |
| 2 | Neural feature extraction — DE + PSD | `data/eeg_features.py` |
| 3 | PECM context sampling + optional RMA utilities | `data/alignment.py` |
| 4 | Neural tower: EEG-Conformer → **v_eeg ∈ R²⁵⁶** | `models/eeg_conformer.py` |
| 5 | Environmental tower: GCN encoder → **v_env ∈ R²⁵⁶** | `models/st_gnn.py` |
| 6 | Bidirectional cross-modal attention → **z_fused ∈ R⁵¹²** | `models/fusion.py` |
| 7 | MLP classifier + composite loss | `models/fusion.py` |
| 8 | GraphLIME attribution + proxy spatial map | `evaluation/attribution.py` |

---

## Repository Structure

```
neuro_geospatial/
├── config.py                        ← All hyperparameters (single source of truth)
├── main.py                          ← Entry point — runs full pipeline
├── requirements.txt
├── README.md
│
├── data/
│   ├── eeg_preprocessing.py         bandpass · ASR · ICA · epoch segmentation
│   ├── eeg_features.py              DE + PSD → X ∈ R^{T×30×10}
│   ├── geospatial.py                10×10 grid · adjacency · LUR integration
│   ├── geo_data.py                  Live API fetchers — OpenAQ · GEE · Overpass
│   ├── alignment.py                 PECM sampling · optional RMA utilities
│   └── lur_model.py                 Land Use Regression for PM2.5 / NO2
│
├── models/
│   ├── eeg_conformer.py             Neural tower (EEG-Conformer)
│   ├── st_gnn.py                    Environmental tower (GCN + projection; optional GRU)
│   ├── fusion.py                    Bidirectional cross-modal attention + loss
│   └── baselines.py                 RF · LSTM · EEG-Conformer-only · Env-MLP
│
├── training/
│   ├── trainer.py                   AdamW · cosine LR · early stopping (patience 15)
│   └── cross_validation.py          Subject-level 5-fold GroupKFold + Dataset classes
│
├── evaluation/
│   ├── metrics.py                   Accuracy · Macro-F1 · ROC-AUC · Wilcoxon
│   ├── attribution.py               GraphLIME perturbation attribution
│   ├── visualization.py             UMAP · proxy spatial map · attribution bars
│   └── gis_visualization.py         GIS maps and spatial indicator plots
│
└── tests/
    ├── test_preprocessing.py
    ├── test_features.py
    ├── test_geospatial.py
    ├── test_no_leakage.py
    ├── test_models.py
    ├── test_metrics_and_visualization.py
    └── test_geo_data_integration.py  Live API integration tests (requires GEE auth)
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/r11up/geocognition.git
cd geocognition
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU note:** The pipeline trains on a single GPU. Install the CUDA-enabled PyTorch build matching your driver:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Authenticate Google Earth Engine (one-time)

Sentinel-2 NDVI and Sentinel-5P NO2/O3 are fetched via the Google Earth Engine Python API. A free GEE account is required.

```bash
earthengine authenticate
```

Follow the browser prompt and note your GEE project ID from [console.cloud.google.com](https://console.cloud.google.com).

---

## Dataset Setup

### EAV Dataset

> Lee, M.-H. et al. (2024). *EAV: EEG-Audio-Video dataset for emotion recognition in conversational contexts.* Scientific Data, 11, 1026.

1. Download: https://doi.org/10.6084/m9.figshare.24633805
2. Extract to `data/EAV/`:
   ```
   data/EAV/
     subject_01/EEG/subject_01.mat
     subject_02/EEG/subject_02.mat
     ...
     subject_42/EEG/subject_42.mat
   ```
3. Set the path via environment variable or `--data_root` flag:
   ```bash
   export EAV_ROOT=data/EAV
   ```

> **MAT v7.3 format:** EAV uses MATLAB v7.3 (HDF5-based) `.mat` files. The loader in `data/eeg_preprocessing.py` automatically falls back to `mat73` → `h5py`. Install one:
> ```bash
> pip install mat73
> ```

> **Note:** Only the five **speaking-condition** trials (label indices 1, 3, 5, 7, 9) are used, corresponding to neutral, sadness, anger, happiness, and relaxed/calm.

### Geospatial Data

All geospatial environmental data is fetched from live APIs when the pipeline runs. No manual download or static geospatial cache is required.

| Source | Data | Study area |
|--------|------|------------|
| OpenAQ v3 | PM2.5 ground measurements | Astana (primary) + Singapore (environmental swap) |
| Google Earth Engine | Sentinel-2 NDVI, Sentinel-5P NO2/O3 | Both cities |
| Overpass API (OSM) | Building footprints → Floor Area Ratio | Both cities |

Production geospatial builders do not load cached feature arrays; they call the live APIs.
An OpenAQ API key is recommended and may be required by current v3 deployments.

---

### Scientific Scope

The current EAV release has no per-epoch GPS/time coordinates. Multimodal training uses label-conditioned PECM environmental priors, while validation/test use marginal live-grid context vectors. These are not measured co-located exposure for each EEG epoch. The `multimodal_oracle_test` CV mode is an explicit upper bound that conditions validation/test context on labels and writes `outputs/table3b_oracle_vs_realistic.csv` during proposed runs.

The `transfer` mode is an environmental-domain swap: it reuses EAV EEG samples and labels while replacing the Astana environmental graph with the Singapore graph. RMA utilities are available in `data/alignment.py`, but the main neural pipeline uses the learned embedding-alignment loss. The RMA ablation fits a frozen per-fold projection target and is not an end-to-end learnable RMA module.

Live geospatial runs write provenance manifests such as `outputs/astana_live_data_manifest.json` and `outputs/singapore_live_data_manifest.json`.

---

## Running the Pipeline

### Full pipeline (all experiments)

```bash
python main.py \
    --mode all \
    --data_root data/EAV \
    --output_dir outputs/ \
    --seed 42 \
    --device cuda \
    --gee_project YOUR_GEE_PROJECT_ID
```

### Individual experiment modes

| Mode | Description |
|------|-------------|
| `all` | All baselines + proposed model + Singapore env-swap + MAUP + interpretation |
| `baselines` | RF, LSTM, EEG-Conformer, Environment-only MLP |
| `proposed` | Unidirectional + bidirectional proposed model (Astana primary) |
| `ablation` | Table 6 ablations: unidirectional attention, no alignment, shuffled labels, GCN-only/no temporal, reversed dose-response, concatenation fusion, InfoNCE alignment, frozen RMA-projected alignment, Env-MLP, EEGNet |
| `transfer` | Singapore environmental-domain swap (requires prior `proposed` run) |
| `maup` | Spatial scale sensitivity analysis |

```bash
# Proposed model only
python main.py --mode proposed --data_root data/EAV --gee_project YOUR_PROJECT_ID

# Baselines only
python main.py --mode baselines --data_root data/EAV --gee_project YOUR_PROJECT_ID

# Singapore environmental-domain swap
python main.py --mode transfer --data_root data/EAV --gee_project YOUR_PROJECT_ID
```

### Configuration

All hyperparameters are set in `config.py`. Key settings:

```python
# Training
CFG.train.n_folds        = 5        # Subject-level cross-validation folds
CFG.train.n_epochs       = 100      # Max epochs per fold
CFG.train.batch_size     = 64
CFG.train.learning_rate  = 1e-4
CFG.train.weight_decay   = 1e-4
CFG.train.patience       = 15       # Early stopping patience

# Model
CFG.model.embed_dim      = 256      # Shared latent dimension
CFG.model.attn_heads     = 8        # Attention heads
CFG.model.lambda_align   = 0.1      # Alignment loss weight λ

# EEG preprocessing
CFG.preprocess.bp_low    = 3.0      # Bandpass low cutoff (Hz)
CFG.preprocess.bp_high   = 50.0     # Bandpass high cutoff (Hz)
CFG.preprocess.target_fs = 100      # Target sampling rate after downsampling
```

---

## Outputs

After running, the `outputs/` directory will contain:

| File | Description |
|------|-------------|
| `table3_astana_results.csv` | Five-fold Astana results for the selected mode, including ablation rows when `--mode ablation` or `--mode all` is used |
| `table3b_oracle_vs_realistic.csv` | Proposed-model realistic vs oracle environmental conditioning comparison |
| `table4_transfer_results.csv` | Singapore environmental-domain swap results |
| `table5_maup.csv` | Spatial scale sensitivity |
| `latent_space_after.pdf/png` | UMAP of shared latent space |
| `spatial_error_map.pdf/png` | 10x10 proxy grid error-risk map |
| `feature_attribution.pdf/png` | GraphLIME per-class attribution bars |
| `figure9_graphlime_environmental_attribution.pdf/png` | GraphLIME attribution summarized across proposed-model CV folds |
| `figure9_graphlime_fold_attributions.npy` | Real per-fold GraphLIME attribution tensor used to generate Figure 9 |
| `proposed_training.pdf/png` | Training curves across folds |
| `checkpoints/fold*_best.pt` | Best model checkpoint per fold |

---

## EEG Cache

Preprocessed EEG and extracted EEG features are cached as `.npy` files in `data/cache/`:

```
data/cache/
  all_epochs.npy              # (16800, 30, 500) float32
  all_labels.npy              # (16800,) int64
  subject_ids.npy             # (16800,) int64
  features_X.npy              # (16800, 30, 10) float32
```

To rerun preprocessing from scratch, delete the cache directory:

```bash
rm -rf data/cache/
```

---

## Tests

```bash
# Unit tests (no network required; API calls are mocked)
python -m pytest tests/ -v -m "not integration" --ignore=tests/test_geo_data_integration.py

# Live API integration tests (requires GEE authentication and OpenAQ access)
GEE_PROJECT=your-project OPENAQ_KEY=your-key \
  python -m pytest tests/test_geo_data_integration.py -v -m integration
```

Model tests require PyTorch. If PyTorch is not installed, `tests/test_models.py`
is skipped by design.

---

## Reproducibility

All random seeds are fixed at `42` for NumPy, PyTorch, and Python random. To reproduce exactly:

```bash
python main.py --mode all --seed 42 --data_root data/EAV --gee_project YOUR_PROJECT_ID
```

> Training on different GPU hardware may produce minor floating-point differences even with fixed seeds due to non-deterministic CUDA operations.

---

## Key Equations

| Equation | Description |
|----------|-------------|
| NDVI | (B8 − B4) / (B8 + B4) |
| L_align | ‖z_EEG − z_ENV‖² |
| DE | h(X) = ½ ln(2πe σ²) |
| w_ij | 1 / (d²_ij + ε) |
| v_eeg | f_Conformer(X; Θ) ∈ R²⁵⁶ |
| GCN | H^{l+1} = σ(Â H^l W^l) |
| v_env | W_pool · mean_pool(W_proj H_GCN) ∈ R²⁵⁶ |
| z_fused | W_f [A_eeg ; A_env] ∈ R⁵¹² |
| L_total | L_CE + λ L_align |

---

## Citation

If you use this code in your research, please cite this paper.

Please also cite the EAV dataset:

```bibtex
@article{lee2024eav,
  title   = {EAV: EEG-Audio-Video dataset for emotion recognition in
             conversational contexts},
  author  = {Lee, Min-Ho and Shomanov, A and Begim, B and others},
  journal = {Scientific Data},
  volume  = {11},
  pages   = {1026},
  year    = {2024},
}
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥ 2.1 | Deep learning framework |
| NumPy | ≥ 1.26 | Numerical arrays |
| SciPy | ≥ 1.11 | Signal processing, linear algebra |
| scikit-learn | ≥ 1.3 | RF baseline, metrics, GroupKFold |
| pandas | ≥ 2.1 | DataFrames, CSV export |
| mat73 / h5py | ≥ 0.62 / 3.9 | EAV .mat v7.3 loading |
| requests | ≥ 2.31 | OpenAQ v3 API |
| earthengine-api | ≥ 0.1.390 | Sentinel-2 NDVI, Sentinel-5P NO2/O3 |
| contextily / pyproj / shapely | ≥ 1.6 / 3.6 / 2.0 | GIS visualisation |
| matplotlib / seaborn | ≥ 3.8 / 0.13 | Plotting |
| umap-learn | ≥ 0.5 | Latent space visualisation (t-SNE fallback if absent) |

---

## Contact

- **Utsav Poudel** — utsav.poudel2021@vitalum.ac.in
- **Jagannath Aryal** — jagannath.aryal@unimelb.edu.au
- **Subramaniyaswamy V** — subramaniyaswamy.v@vit.ac.in

## License

This repository is released under the MIT License. See [LICENSE](./LICENSE).
