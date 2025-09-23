# HGSOC Drug Delivery Baseline

## Problem Statement

High-Grade Serous Ovarian Cancer (HGSOC) presents unique drug delivery challenges due to:
- **Heterogeneous tumor perfusion** across different anatomical sites
- **Variable drug clearance** based on patient physiology and co-medications  
- **Stage-dependent accessibility** affecting therapeutic concentrations
- **Limited biomarker stratification** for personalized dosing

This project develops a **synthetic pharmacokinetic/pharmacodynamic (PK/PD) modeling framework** to explore drug delivery optimization strategies for HGSOC patients. We simulate realistic patient cohorts and evaluate machine learning approaches for predicting treatment response based on PK/PD parameters, tumor characteristics, and patient factors.

## Data & Methodology

**Synthetic Data Approach**: This project uses **entirely synthetic data** to ensure reproducibility and avoid privacy constraints while maintaining clinical realism. Our simulation incorporates:

- **Patient demographics**: Age, weight, organ function (liver/kidney)
- **Tumor characteristics**: Stage (I-IV), size, location, perfusion heterogeneity
- **PK parameters**: Clearance, volume of distribution, protein binding
- **PD relationships**: Dose-response curves, resistance mechanisms
- **Treatment outcomes**: Binary response classification with noise

**Modeling Pipeline**: Dual approach comparing interpretable (logistic regression) vs. complex (gradient boosting) models with comprehensive evaluation including calibration, fairness, and sensitivity analysis.

> **Capabilities**: deterministic cohort simulation, feature engineering with scaling + interactions, dual-model training (logreg & HGB), calibration/Brier/ROC/PR diagnostics, subgroup fairness tables, PK prior stress tests, optional Streamlit scenario explorer.
>
> **Not in scope**: clinical recommendations, mechanistic dosing, real patient data, uncertainty quantification beyond simple bands, EHR integration.

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

## Roadmap

### Phase 1: Foundation 
- [x] Synthetic cohort simulation with realistic PK/PD parameters
- [x] Dual-model training (logistic regression + gradient boosting)
- [x] Comprehensive model evaluation (calibration, ROC/PR, Brier)
- [x] Subgroup fairness analysis by tumor stage and size
- [x] PK sensitivity analysis for robustness testing
- [x] Interactive Streamlit demo for scenario exploration

### Phase 2: Enhanced Modeling (Planned)
- [ ] **Multi-target optimization**: Efficacy vs. toxicity trade-offs
- [ ] **Temporal modeling**: PK/PD dynamics over treatment cycles
- [ ] **Bayesian uncertainty**: Credible intervals for risk predictions
- [ ] **Feature attribution**: SHAP analysis for model interpretability
- [ ] **Cross-validation**: Temporal splits simulating real deployment

### Phase 3: Clinical Translation (Future)
- [ ] **Domain expert validation**: Oncology consultation on simulation realism
- [ ] **Regulatory preparation**: Documentation for FDA/EMA submission pathways
- [ ] **Real-world integration**: APIs for EHR systems and clinical decision support
- [ ] **Federated learning**: Multi-site model training while preserving privacy

## Responsible use

This repository is intended for interview-style discussions and rapid prototyping. No real patients, dosing, or clinical outcomes are represented. Always verify assumptions with domain experts before carrying insights forward.
