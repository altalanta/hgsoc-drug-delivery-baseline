#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def score_targets(cohort: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    ann = annotations.copy()
    ann['is_surface'] = ann['is_surface'].astype(int)

    # Build long-form matrix of expression
    expr_cols = [c for c in cohort.columns if c.startswith('expr_')]
    long = cohort[['patient_id'] + expr_cols].melt(id_vars=['patient_id'], var_name='feature', value_name='expr')
    long['gene'] = long['feature'].str.replace('expr_', '', regex=False)
    long = long.merge(ann, on='gene', how='left')

    # Tumor-to-normal contrast proxy: expr / (normal_ovary_expr + 0.1)
    long['t_n_contrast'] = long['expr'] / (long['normal_ovary_expr'] + 0.1)

    # Internalization weight
    rate_w = long['internalization_rate'].map({'fast': 1.0, 'moderate': 0.7, 'slow': 0.4, 'NA': 0.5}).fillna(0.5)

    # Composite target score per patient-gene
    long['target_score'] = (
        0.5 * np.log1p(long['expr']) +
        0.3 * np.log1p(long['t_n_contrast']) +
        0.2 * rate_w
    ) * (0.9 + 0.1 * long['is_surface'])  # small boost if surface-validated

    # Summaries
    per_patient = long.groupby(['patient_id', 'gene'], as_index=False).agg(
        expr=('expr', 'median'),
        t_n_contrast=('t_n_contrast', 'median'),
        target_score=('target_score', 'median')
    )

    population = per_patient.groupby('gene', as_index=False).agg(
        median_score=('target_score', 'median'),
        pct_high=('target_score', lambda x: float((x > x.median()).mean()))
    ).sort_values(['median_score', 'pct_high'], ascending=False)

    return per_patient, population


def main():
    ap = argparse.ArgumentParser(description='Target prioritization for HGSOC')
    ap.add_argument('--cohort', required=True)
    ap.add_argument('--annotations', required=True)
    ap.add_argument('--outdir', default='outputs')
    args = ap.parse_args()

    cohort = pd.read_csv(args.cohort)
    ann = pd.read_csv(args.annotations)
    per_patient, population = score_targets(cohort, ann)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    per_patient.to_csv(out / 'per_patient_target_scores.csv', index=False)
    population.to_csv(out / 'population_target_ranking.csv', index=False)
    print(f"Saved per-patient and population rankings to {out}")


if __name__ == '__main__':
    main()

