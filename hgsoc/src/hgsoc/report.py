"""Generate Markdown report for HGSOC baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render markdown report from evaluation metrics.")
    parser.add_argument("--metrics", type=Path, required=True, help="metrics.json produced by eval")
    parser.add_argument("--out", type=Path, required=True, help="Output markdown path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = json.loads(args.metrics.read_text())
    report = f"""# HGSOC Drug Delivery Baseline

> **Synthetic educational demo** — do not use for clinical decisions.

## Summary metrics

| Model | ROC AUC | PR AUC | Brier |
|-------|---------|--------|-------|
| Logistic Regression | {metrics['logreg_roc_auc']:.3f} | {metrics['logreg_pr_auc']:.3f} | {metrics['logreg_brier']:.3f} |
| HistGradientBoosting | {metrics['hgb_roc_auc']:.3f} | {metrics['hgb_pr_auc']:.3f} | {metrics['hgb_brier']:.3f} |

Calibration slope={metrics['calibration_slope']:.3f}, intercept={metrics['calibration_intercept']:.3f}.

## Fairness snapshot

- Subgroup Brier gap (stage/size): {metrics['subgroup_brier_gap']:.3f}

See `reports/subgroup_table.csv` for per-group details.

## PK prior sensitivity

- Brier Δ (clearance −10%): {metrics['pk_prior_minus10_delta']:.3f}
- Brier Δ (clearance +10%): {metrics['pk_prior_plus10_delta']:.3f}

## Figures

![Calibration](calibration.png)

![ROC/PR](roc_pr.png)

## Limits

- Synthetic cohort; no validation on real HGSOC patients.
- Simplified PK/PD relationships; no inter-patient covariates beyond the simulated features.
- HistGradientBoosting is not calibrated for dosing recommendations; use temperature scaling for deployment.

## Next steps

1. Introduce external validation cohorts with heterogeneous imaging/PK priors.
2. Add Bayesian calibration and probabilistic sensitivity analysis for dosing.
3. Extend Streamlit app to explore subgroup explanations and uncertainty bands.
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)


if __name__ == "__main__":
    main()
