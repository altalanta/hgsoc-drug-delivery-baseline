#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate_cohort(n=100, seed=13, annotations_path="data/annotations/surface_targets.csv"):
    rng = np.random.default_rng(seed)
    ann = pd.read_csv(annotations_path)
    targets = ann['gene'].tolist()

    patients = []
    for i in range(n):
        pid = f"P{i:04d}"
        age = int(rng.normal(62, 7))
        stage = rng.choice(['IIIC', 'IV'])
        brca = rng.choice(['WT', 'BRCA1', 'BRCA2'], p=[0.78, 0.14, 0.08])
        hrd = float(np.clip(rng.normal(45 if brca != 'WT' else 32, 10), 0, 100))
        platinum = rng.choice(['Sensitive', 'Resistant'], p=[0.65, 0.35])
        tumor_burden_idx = float(np.clip(rng.normal(0.6, 0.2), 0.05, 1.0))
        ascites_ml = int(np.clip(rng.normal(1200, 500), 0, 5000))
        ip_feasible = bool(ascites_ml > 200)  # simplistic proxy for IP delivery feasibility
        ca125 = int(np.clip(rng.normal(750, 300), 35, 5000))
        vascular_score = float(np.clip(rng.normal(0.55, 0.15), 0, 1))
        stroma_score = float(np.clip(rng.normal(0.5, 0.2), 0, 1))
        immune_score = float(np.clip(rng.normal(0.35, 0.2), 0, 1))

        row = {
            'patient_id': pid,
            'age': age,
            'stage': stage,
            'brca_status': brca,
            'hrd_score': hrd,
            'platinum_status': platinum,
            'tumor_burden_index': tumor_burden_idx,
            'ascites_ml': ascites_ml,
            'ip_route_feasible': ip_feasible,
            'ca125': ca125,
            'vascular_score': vascular_score,
            'stroma_score': stroma_score,
            'immune_score': immune_score,
        }

        # Expression of targets scaled by annotation priors + patient variability
        for _, a in ann.iterrows():
            mu = float(a['tumor_expr_scale'])
            # add patient-specific variation and correlation with HRD for DNA-repair linked receptors (toy)
            expr = np.clip(rng.lognormal(mean=np.log(mu), sigma=0.25), 0.1, 10)
            row[f"expr_{a['gene']}"] = expr

        patients.append(row)

    return pd.DataFrame(patients)


def main():
    ap = argparse.ArgumentParser(description='Generate a synthetic HGSOC cohort')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=13)
    ap.add_argument('--annotations', type=str, default='data/annotations/surface_targets.csv')
    ap.add_argument('--out', type=str, default='data/synthetic_hgsoc_cohort.csv')
    args = ap.parse_args()

    df = generate_cohort(n=args.n, seed=args.seed, annotations_path=args.annotations)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} patients to {out}")


if __name__ == '__main__':
    main()

