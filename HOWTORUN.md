# How to Run

This project runs the neuro-geospatial EEG pipeline with live geospatial APIs:
- OpenAQ v3 for PM2.5
- Google Earth Engine for Sentinel-2 NDVI and Sentinel-5P NO2/O3
- Overpass API for OpenStreetMap building footprints and FAR

Production geospatial features are fetched from the live data providers for each run.

## Requirements

- Python 3.10+
- EAV dataset from the original creators
- Google Earth Engine account and authentication
- OpenAQ v3 API key, recommended and may be required
- SEED-IV dataset if you want the paper-aligned EEG-Conformer pretraining

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
earthengine authenticate
```

Set paths and API credentials:

```bash
export EAV_ROOT=/path/to/eav_dataset
export GEE_PROJECT=your-gee-project
export OPENAQ_KEY=your-openaq-key
```

## EAV Layout

```text
data/eav_dataset/
  subject_01/EEG/subject_01.mat
  subject_02/EEG/subject_02.mat
  ...
  subject_42/EEG/subject_42.mat
```

## SEED-IV Pretraining

The paper-aligned run initializes the EEG-Conformer encoder from SEED-IV.

```bash
python scripts/pretrain_seediv.py \
  --seediv_root data/seed_iv \
  --output_ckpt checkpoints/eeg_conformer_seediv.pt \
  --device cuda \
  --epochs 50
```

Use the checkpoint in `main.py`:

```bash
export SEEDIV_CKPT=checkpoints/eeg_conformer_seediv.pt
```

or pass it explicitly with `--seediv_ckpt`. If the checkpoint is absent, the
pipeline logs one warning and runs with random EEG encoder initialization.

## PECM Sampling Policy

The code follows the paper-aligned supervised PECM policy:

| Split | Method | Label access |
|---|---|---|
| Train | `pecm.sample(y_train)` | Yes, supervised prior |
| Validation | `pecm.sample_unconditional(n)` | No |
| Test | `pecm.sample_unconditional(n)` | No |
| Environment-only baseline | `pecm.sample(y_train)` for train, `sample_unconditional(n)` for validation/test | Train only |
| Oracle comparison | `pecm.sample(y)` for train/validation/test | Yes, upper bound only |

The EAV release has no per-epoch GPS/time coordinates, so these vectors are
PECM environmental context priors, not measured co-located exposure.

## Run Modes

```bash
# Full pipeline: baselines, proposed, ablations, env-swap, MAUP, interpretation
python main.py --mode all \
  --data_root "$EAV_ROOT" \
  --device cuda \
  --gee_project "$GEE_PROJECT" \
  --openaq_key "$OPENAQ_KEY" \
  --seediv_ckpt checkpoints/eeg_conformer_seediv.pt \
  --year 2025

# Baselines only
python main.py --mode baselines --data_root "$EAV_ROOT" \
  --gee_project "$GEE_PROJECT" --openaq_key "$OPENAQ_KEY"

# Proposed Astana model only
python main.py --mode proposed --data_root "$EAV_ROOT" \
  --gee_project "$GEE_PROJECT" --openaq_key "$OPENAQ_KEY" \
  --seediv_ckpt checkpoints/eeg_conformer_seediv.pt

# Table 6 ablations
python main.py --mode ablation --data_root "$EAV_ROOT" \
  --gee_project "$GEE_PROJECT" --openaq_key "$OPENAQ_KEY" \
  --seediv_ckpt checkpoints/eeg_conformer_seediv.pt

# Singapore environmental-domain swap, requires prior Astana checkpoint
python main.py --mode transfer --data_root "$EAV_ROOT" \
  --gee_project "$GEE_PROJECT" --openaq_key "$OPENAQ_KEY"

# MAUP sensitivity
python main.py --mode maup --data_root "$EAV_ROOT" \
  --gee_project "$GEE_PROJECT" --openaq_key "$OPENAQ_KEY"
```

The ablation mode runs the unidirectional attention, no-alignment, shuffled-label,
GCN-only/no-temporal, reversed dose-response, concatenation-fusion, InfoNCE,
frozen RMA-projected alignment, Env-MLP, and EEGNet rows. The RMA row uses a
per-fold frozen projection target from `data/alignment.py`; it is not an
end-to-end learnable RMA module.

The `transfer` mode is an environmental-domain swap. It reuses EAV EEG and
labels while replacing the Astana environmental graph with Singapore's graph;
it is not Singapore EEG transfer.

## GIS Figures

```bash
python scripts/run_gis_visualization.py \
  --gee_project "$GEE_PROJECT" \
  --openaq_key "$OPENAQ_KEY"
```

Figure 8 is a proxy environmental error-risk map unless real per-epoch node
IDs are supplied elsewhere.

## Tests

```bash
# Unit tests, no live network
python -m pytest tests/ -v --ignore=tests/test_geo_data_integration.py

# Live API integration tests
GEE_PROJECT="$GEE_PROJECT" OPENAQ_KEY="$OPENAQ_KEY" \
  python -m pytest tests/test_geo_data_integration.py -v -m integration
```

## Outputs

- `outputs/table3_astana_results.csv` (selected Astana baseline/proposed/ablation rows)
- `outputs/table3b_oracle_vs_realistic.csv`
- `outputs/table4_transfer_results.csv`
- `outputs/table5_maup.csv`
- `outputs/*_live_data_manifest.json`
- `outputs/*.pdf` and `outputs/*.png`
- `checkpoints/proposed_astana_best.pt`
- `checkpoints/eeg_conformer_seediv.pt`

OpenAQ station counts, LUR R2/RMSE, and live acquisition settings are logged at
runtime. Live geospatial manifests in `outputs/` record bbox, year, API source,
and acquisition window settings.
