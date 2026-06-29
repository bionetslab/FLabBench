# FLabBench

FLabBench: A Large-Scale Benchmark Evaluating the Impact of Cohort Characteristics on Predictive Model Performance Across Over 4,000 Clinical Cohorts.

---

## Overview

FLabBench runs in three stages:

```
1. Cohort Extraction   →   2. Feature Extraction   →   3. Model Training
flab_cohorts/              flab_features/               flab_training/
```

- **Cohort extraction**: builds patient cohorts from MIMIC-IV using either disease trajectory (DTB) or literature-based (LIT) definitions, with survival labels (D1/D2 endpoints)
- **Feature extraction**: extracts longitudinal lab time series for each cohort within a configurable time window before discharge
- **Model training**: trains and evaluates time-series models (GRU, LSTM, STraTS, SAND, EMIT, TCN, GRU-D, InterpNet) and ML models (Random Forest, XGBoost, LightGBM, CatBoost) with nested cross-validation

---

## Setup

```bash
conda env create -f environment.yml        # Linux / HPC
conda env create -f environment_mac.yml    # macOS
conda activate flabbench
```

Set the MIMIC-IV data path:

```bash
export MIMIC_IV_PATH="/path/to/mimiciv/2.0/"
```

---

## Usage

### 1. Cohort Extraction

```bash
# Literature-based cohorts
python -m flab_cohorts.extract_cohorts --extractor LIT --cohort aki
python -m flab_cohorts.extract_cohorts --extractor LIT --cohort all

# Disease trajectory cohorts (ICD-10 based)
python -m flab_cohorts.extract_cohorts --extractor DTB --cohort A08-A41
python -m flab_cohorts.extract_cohorts --extractor DTB --cohort DTB_all
```

**Extractors:**
- `DTB` — disease trajectory-based: builds cohorts from ICD-10 code pairs (e.g. A08→A41), supports ~4000 cohorts
- `LIT` — literature-based: predefined cohorts matching published survival prediction studies (neutropenic fever, AKI, heart failure, pneumonia, etc.)

---

### 2. Feature Extraction

```bash
python -m flab_features.extract_features --extractor LIT --cohort neutropenic_fever
python -m flab_features.extract_features --extractor DTB --cohort A08-A41
python -m flab_features.extract_features --extractor LIT --cohort all --days 14
```

| Argument | Default | Description |
|---|---|---|
| `--days` | 14 | Days before discharge to extract labs |
| `--feature-selection` | True | Filter to global top-100 features |

---

### 3. Model Training

**Time-series models:**

```bash
python -m flab_training.main \
    --extractor DTB \
    --cohort A08-A41 \
    --model_type strats \
    --fold 0 \
    --days_before_discharge 14
```

**ML models:**

```bash
bash run_ml.sh
# or directly:
python test_run.py --cohort neutropenic_fever --model_type random_forest --fold 0
```

**Available models:**

| Type | Models |
|---|---|
| Time-series | `gru`, `lstm`, `strats`, `sand`, `emit`, `tcn`, `grud`, `interpnet`, `mlp` |
| ML | `random_forest`, `xgboost`, `lightgbm`, `catboost` |

---

## Project Structure

```
FLabBench-pipeline/
├── flab_cohorts/
│   ├── extract_cohorts.py          # entry point for cohort extraction
│   ├── extractors/
│   │   ├── DTB_extractor.py        # disease trajectory cohorts
│   │   ├── LIT_extractor.py        # literature-based cohorts
│   │   └── LIT/                    # per-condition cohort definitions
│   └── utils/
├── flab_features/
│   ├── extract_features.py         # entry point for feature extraction
│   └── feature_extractor.py
├── flab_training/
│   ├── main.py                     # time-series model training
│   ├── ts_models/                  # GRU, LSTM, STraTS, SAND, EMIT, TCN, GRU-D, InterpNet
│   ├── config_files/               # model hyperparameter configs
│   └── envmanager.py               # training orchestration
├── config/
│   └── constants.py                # paths, ICD chapter mapping, seeds
├── analysis/
│   ├── analyze_cohorts.ipynb       # cohort and feature prevalence analysis
│   └── analyze_results.ipynb       # results analysis
└── data/
    └── top_features/
        └── mimic_top100_features.pkl   # global top-100 lab features
```

---

## Data

Requires [MIMIC-IV v2.0](https://physionet.org/content/mimiciv/2.0/) (credentialed access via PhysioNet).

Key tables used: `admissions`, `patients`, `diagnoses_icd`, `labevents`, `d_labitems`.

---
