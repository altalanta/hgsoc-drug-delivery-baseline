"""Synthetic cohort simulation for HGSOC drug delivery baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def simulate_cohort(n: int, seed: int = 1337) -> pd.DataFrame:
    """Simulate a pharmacology cohort with PK/PD covariates."""

    rng = np.random.default_rng(seed)
    ligand_density = rng.lognormal(mean=1.2, sigma=0.35, size=n)
    tumor_size_cm = rng.gamma(shape=3.0, scale=1.2, size=n)
    perfusion_idx = rng.uniform(0.3, 1.0, size=n)
    pk_clearance = rng.normal(loc=0.8, scale=0.12, size=n)
    pk_vd = rng.normal(loc=3.2, scale=0.5, size=n)
    pd_sensitivity = rng.normal(loc=0.0, scale=1.0, size=n)
    stage_probs = np.clip(perfusion_idx * 0.2 + tumor_size_cm / 10, 0, 1)
    stage = np.where(stage_probs < 0.25, "I", np.where(stage_probs < 0.5, "II", np.where(stage_probs < 0.75, "III", "IV")))

    exposure_auc = ligand_density * perfusion_idx * pk_vd / np.maximum(pk_clearance, 1e-3)
    exposure_auc = np.clip(exposure_auc, 0, np.quantile(exposure_auc, 0.995))
    normalized_auc = (exposure_auc - exposure_auc.mean()) / (exposure_auc.std() + 1e-6)

    logit = (
        -1.8
        + 0.45 * normalized_auc
        + 0.6 * perfusion_idx
        - 0.18 * tumor_size_cm
        + 0.35 * pd_sensitivity
    )
    response_prob = 1 / (1 + np.exp(-logit))
    response_binary = rng.binomial(1, response_prob)
    response_continuous = np.clip(
        100 * (response_prob + rng.normal(0, 0.08, size=n)),
        0,
        100,
    )

    df = pd.DataFrame(
        {
            "ligand_density": ligand_density,
            "tumor_size_cm": tumor_size_cm,
            "perfusion_idx": perfusion_idx,
            "pk_clearance": pk_clearance,
            "pk_vd": pk_vd,
            "pd_sensitivity": pd_sensitivity,
            "exposure_auc": exposure_auc,
            "stage": stage,
            "response_binary": response_binary,
            "response_continuous": response_continuous,
        }
    )
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate synthetic HGSOC drug delivery cohort.")
    parser.add_argument("--n", type=int, default=500, help="Number of patients")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--out", type=Path, required=True, help="Output parquet path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = simulate_cohort(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)


if __name__ == "__main__":
    main()
