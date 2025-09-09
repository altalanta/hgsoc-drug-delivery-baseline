#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from modeling.pkpd_delivery_sim import PKParams, simulate, kpi


def select_route(cohort: pd.DataFrame) -> str:
    # Heuristic: if majority has IP feasible, prefer IP for locoregional delivery
    frac_ip = cohort['ip_route_feasible'].mean()
    return 'IP' if frac_ip > 0.5 else 'IV'


def modality_for_target(annotations: pd.DataFrame, target: str) -> str:
    row = annotations.loc[annotations['gene'] == target]
    if row.empty:
        return 'nanoparticle'
    rate = str(row.iloc[0]['internalization_rate']).lower()
    # Favor ADC/nanoparticle if internalization is moderate/fast; fall back to small molecule otherwise
    return 'nanoparticle' if rate in ('fast', 'moderate') else 'small_molecule'


def sweep_design(route: str, base_modality: str, doses=(50, 100), k_rel_grid=(0.03, 0.1, 0.3)):
    rows = []
    for dose in doses:
        if base_modality == 'small_molecule':
            prof = simulate(route=route, modality='small_molecule', dose_mg=dose)
            met = kpi(prof)
            rows.append({**met, 'route': route, 'modality': 'small_molecule', 'dose_mg': dose, 'k_rel': 0.0})
        else:
            for krel in k_rel_grid:
                prof = simulate(route=route, modality='nanoparticle', dose_mg=dose, params=PKParams(k_rel=krel))
                met = kpi(prof)
                rows.append({**met, 'route': route, 'modality': 'nanoparticle', 'dose_mg': dose, 'k_rel': krel})
    return pd.DataFrame(rows)


def plot_tradeoff(df: pd.DataFrame, outdir: Path, title: str):
    fig, ax = plt.subplots(figsize=(7,5))
    for (mod, route), grp in df.groupby(['modality','route']):
        ax.scatter(grp['AUC_plasma'], grp['AUC_tumor'], label=f"{mod}/{route}")
        for _, r in grp.iterrows():
            ax.annotate(f"{int(r['dose_mg'])}mg,k={r['k_rel']}", (r['AUC_plasma'], r['AUC_tumor']), fontsize=8)
    ax.set_xlabel('AUC Plasma')
    ax.set_ylabel('AUC Tumor')
    ax.set_title(title)
    ax.legend()
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / 'design_tradeoff.png'
    fig.tight_layout()
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


def main():
    ap = argparse.ArgumentParser(description='Design selection for HGSOC delivery')
    ap.add_argument('--cohort', required=True)
    ap.add_argument('--annotations', required=True)
    ap.add_argument('--target', default='FOLR1')
    ap.add_argument('--route', choices=['auto','IV','IP'], default='auto')
    ap.add_argument('--outdir', default='outputs')
    args = ap.parse_args()

    cohort = pd.read_csv(args.cohort)
    ann = pd.read_csv(args.annotations)

    route = select_route(cohort) if args.route == 'auto' else args.route
    modality = modality_for_target(ann, args.target)

    results = []
    # Compare both routes and modalities for completeness
    for rt in ['IV', 'IP']:
        results.append(sweep_design(rt, 'small_molecule'))
        results.append(sweep_design(rt, 'nanoparticle'))
    res = pd.concat(results, ignore_index=True)

    # Rank by tumor AUC high and plasma AUC low (composite)
    res['score'] = res['AUC_tumor'] / (np.sqrt(res['AUC_plasma']) + 1e-6)
    res = res.sort_values('score', ascending=False)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / 'design_candidates.csv', index=False)
    plot_tradeoff(res, out, title=f"Design Tradeoffs for {args.target}")

    print(f"Target: {args.target} | Suggested route: {route} | Base modality: {modality}")
    print(f"Top candidates saved to {out / 'design_candidates.csv'}")


if __name__ == '__main__':
    main()

