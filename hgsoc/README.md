# HGSOC Drug Delivery Baseline

Synthetic pharmacokinetic/pharmacodynamic (PK/PD) playground for High-Grade Serous Ovarian Cancer. The project simulates a cohort, fits interpretable (logistic regression) and non-linear (Histogram Gradient Boosting) models, and reports calibration, subgroup fairness, and PK prior sensitivity.

> ⚙️ **Capabilities**: deterministic cohort simulation, feature engineering with scaling + interactions, dual-model training (logreg & HGB), calibration/Brier/ROC/PR diagnostics, subgroup fairness tables, PK prior stress tests, optional Streamlit scenario explorer.
>
> 🚫 **Not in scope**: clinical recommendations, mechanistic dosing, real patient data, uncertainty quantification beyond simple bands, EHR integration.

## Quickstart

```bash
make env        # create virtualenv + install deps
make simulate   # generate synthetic cohort (parquet)
make train      # train logistic + HGB models, save to artifacts/
make eval       # produce metrics, calibration/ROC/PR plots, fairness tables
make report     # render markdown summary in reports/README.md
```

All steps complete on CPU within a few minutes for `n=500` subjects.

## Layout

```
hgsoc/
├── src/hgsoc/        # simulation, features, models, eval, report, streamlit app
├── artifacts/        # trained models + feature spec (checked in for demo)
├── reports/          # metrics.json, calibration.png, roc_pr.png, subgroup_table.csv, README.md
├── notebooks/        # lightweight PK sensitivity scratch notebook
└── tests/            # pytest smoke tests for simulation + calibration
```

## Sample metrics (from `reports/metrics.json`)

| Model | ROC AUC | PR AUC | Brier |
|-------|---------|--------|-------|
| Logistic Regression | 0.760 | 0.357 | 0.204 |
| HistGradientBoosting | 0.943 | 0.907 | 0.036 |

- Calibration slope ≈ 1.371, intercept ≈ 0.332
- Subgroup Brier gap (stage / size splits): 0.058
- PK prior stress (clearance ±10%): ΔBrier of +0.024 / +0.027

## Artifacts

Pre-generated diagnostics are committed for reviewers:

- `reports/calibration.png` – reliability curve (10-bin).
- `reports/roc_pr.png` – ROC + PR charts for the HGB model.
- `reports/subgroup_table.csv` – Brier scores by stage and tumor size terciles.
- `reports/metrics.json` & `reports/README.md` – machine-readable metrics + lite model card.

## Streamlit demo

Optional UI for rapid “what-if” experimentation:

```bash
streamlit run src/hgsoc/app.py
```

Sliders control ligand density, tumor size, perfusion, PK clearance/volume, PD sensitivity, and stage. Predictions from both models plus an approximate confidence band are displayed alongside the hold-out calibration curve.

## Responsible use

This repository is intended for interview-style discussions and rapid prototyping. No real patients, dosing, or clinical outcomes are represented. Always verify assumptions with domain experts before carrying insights forward.
