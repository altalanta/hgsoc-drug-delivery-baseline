"""Evaluation utilities for HGSOC baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .features import FeatureSpec, transform_features


def _load_artifacts(models_dir: Path):
    logreg = joblib.load(models_dir / "logreg.joblib")
    hgb = joblib.load(models_dir / "gbm.joblib")
    spec: FeatureSpec = joblib.load(models_dir / "feature_spec.joblib")
    return logreg, hgb, spec


def _calibration_params(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    eps = 1e-6
    logits = np.log(np.clip(probs, eps, 1 - eps) / np.clip(1 - probs, eps, 1 - eps))
    logits = logits.reshape(-1, 1)
    clf = LogisticRegression(max_iter=1000, penalty=None, solver="lbfgs").fit(logits, y_true)
    return {"intercept": float(clf.intercept_[0]), "slope": float(clf.coef_[0][0])}


def _subgroup_brier(df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    df_local = df.copy()
    df_local["prob"] = probs
    df_local["tumor_size_bin"] = pd.qcut(df_local["tumor_size_cm"], q=3, labels=["small", "medium", "large"])
    rows = []
    for group_name, group_df in [
        ("stage", df_local.groupby("stage")),
        ("tumor_size", df_local.groupby("tumor_size_bin")),
    ]:
        for label, part in group_df:
            rows.append(
                {
                    "group": group_name,
                    "category": label,
                    "count": len(part),
                    "brier": brier_score_loss(part["response_binary"], part["prob"]),
                }
            )
    return pd.DataFrame(rows)


def _pk_sensitivity(
    df: pd.DataFrame,
    spec: FeatureSpec,
    model,
    shift: float,
) -> float:
    shifted = df.copy()
    shifted["pk_clearance"] *= shift
    X_shifted = transform_features(shifted, spec)
    probs = model.predict_proba(X_shifted)[:, 1]
    return float(brier_score_loss(shifted["response_binary"], probs))


def evaluate_models(
    data_path: Path,
    models_dir: Path,
    out_dir: Path,
) -> Dict[str, float]:
    df = pd.read_parquet(data_path)
    logreg, hgb, spec = _load_artifacts(models_dir)
    X = transform_features(df, spec)
    y_true = df["response_binary"].to_numpy()

    probs_logreg = logreg.predict_proba(X)[:, 1]
    probs_hgb = hgb.predict_proba(X)[:, 1]

    metrics = {
        "logreg_roc_auc": float(roc_auc_score(y_true, probs_logreg)),
        "logreg_pr_auc": float(average_precision_score(y_true, probs_logreg)),
        "logreg_brier": float(brier_score_loss(y_true, probs_logreg)),
        "hgb_roc_auc": float(roc_auc_score(y_true, probs_hgb)),
        "hgb_pr_auc": float(average_precision_score(y_true, probs_hgb)),
        "hgb_brier": float(brier_score_loss(y_true, probs_hgb)),
    }

    calib = _calibration_params(y_true, probs_hgb)
    metrics.update({f"calibration_{k}": v for k, v in calib.items()})

    frac_pos, mean_pred = calibration_curve(y_true, probs_hgb, n_bins=10)
    roc_fpr, roc_tpr, _ = roc_curve(y_true, probs_hgb)
    pr_precision, pr_recall, _ = precision_recall_curve(y_true, probs_hgb)

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.plot(mean_pred, frac_pos, marker="o", label="HGB")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curve")
    plt.tight_layout()
    plt.savefig(out_dir / "calibration.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(roc_fpr, roc_tpr, label=f"ROC AUC={metrics['hgb_roc_auc']:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC - HGB")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(pr_recall, pr_precision, label=f"PR AUC={metrics['hgb_pr_auc']:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR - HGB")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_pr.png", dpi=200)
    plt.close()

    subgroup = _subgroup_brier(df, probs_hgb)
    subgroup.to_csv(out_dir / "subgroup_table.csv", index=False)
    metrics["subgroup_brier_gap"] = float(subgroup.groupby("group")["brier"].apply(lambda s: s.max() - s.min()).max())

    base_brier = metrics["hgb_brier"]
    minus = _pk_sensitivity(df, spec, hgb, 0.9)
    plus = _pk_sensitivity(df, spec, hgb, 1.1)
    metrics["pk_prior_minus10_delta"] = float(minus - base_brier)
    metrics["pk_prior_plus10_delta"] = float(plus - base_brier)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained HGSOC models.")
    parser.add_argument("--data", type=Path, required=True, help="Parquet data file")
    parser.add_argument("--models", type=Path, required=True, help="Artifacts directory")
    parser.add_argument("--out", type=Path, required=True, help="Report directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_models(args.data, args.models, args.out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
