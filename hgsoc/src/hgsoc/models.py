"""Model training for HGSOC baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .features import DatasetBundle, FeatureSpec, prepare_features


def _validation_metrics(bundle: DatasetBundle, logreg, hgb) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for name, model in {"logreg": logreg, "hgb": hgb}.items():
        probs = model.predict_proba(bundle.X_valid)[:, 1]
        metrics[f"{name}_roc_auc"] = float(roc_auc_score(bundle.y_valid, probs))
        metrics[f"{name}_pr_auc"] = float(average_precision_score(bundle.y_valid, probs))
        metrics[f"{name}_brier"] = float(brier_score_loss(bundle.y_valid, probs))
    return metrics


def train_models(data_path: Path, out_dir: Path) -> Dict[str, float]:
    df = pd.read_parquet(data_path)
    bundle = prepare_features(df)

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(bundle.X_train, bundle.y_train)

    hgb = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.08,
        max_iter=200,
        l2_regularization=0.01,
        random_state=42,
    )
    hgb.fit(bundle.X_train, bundle.y_train)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(logreg, out_dir / "logreg.joblib")
    joblib.dump(hgb, out_dir / "gbm.joblib")
    joblib.dump(bundle.spec, out_dir / "feature_spec.joblib")

    metrics = _validation_metrics(bundle, logreg, hgb)
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train logistic + HGB models on synthetic cohort.")
    parser.add_argument("--data", type=Path, required=True, help="Input parquet file")
    parser.add_argument("--out", type=Path, required=True, help="Output artifact directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_models(args.data, args.out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
