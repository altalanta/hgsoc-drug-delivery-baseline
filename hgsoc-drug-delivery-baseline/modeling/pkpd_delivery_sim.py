#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass
class PKParams:
    # Volumes (L)
    Vp: float = 3.0   # plasma volume
    Vi: float = 0.5   # peritoneal cavity effective volume
    Vt: float = 1.0   # tumor interstitium effective volume

    # Clearances / transfer rates (1/h)
    kel_p: float = 0.2            # plasma elimination
    k_pi: float = 0.15            # peritoneal -> plasma absorption
    k_iv: float = 0.0             # IV bolus input (handled as dose at t=0)
    k_p_t: float = 0.05           # plasma -> tumor transfer
    k_t_p: float = 0.02           # tumor -> plasma return
    k_i_t: float = 0.03           # peritoneal -> tumor transfer (IP advantage)

    # Release rate (1/h) for nanoparticle payload
    k_rel: float = 0.1


def pk_ode(t, y, p: PKParams, route: str, modality: str):
    # y = [Cp_free, Ci_free, Ct_free, Cp_rel, Ci_rel, Ct_rel] for nanoparticle payload
    # For small molecule, rel pools are zero and k_rel ignored.
    Cp, Ci, Ct, Cpr, Cir, Ctr = y

    # Effective transfer rates differ by modality (nanoparticles slower extravasation but better peritoneal retention)
    if modality == 'nanoparticle':
        k_p_t = p.k_p_t * 0.6
        k_t_p = p.k_t_p * 0.6
        k_i_t = p.k_i_t * 1.2
        kel_p = p.kel_p * 0.6  # slower clearance
        k_rel = p.k_rel
    else:
        k_p_t = p.k_p_t
        k_t_p = p.k_t_p
        k_i_t = p.k_i_t * 0.7
        kel_p = p.kel_p
        k_rel = 0.0

    # Free pools dynamics
    dCp = -kel_p*Cp - k_p_t*Cp + k_t_p*Ct + p.k_pi*Ci
    dCi = -p.k_pi*Ci - k_i_t*Ci
    dCt = k_p_t*Cp + k_i_t*Ci - k_t_p*Ct

    # Released payload from nanoparticle
    dCpr = k_rel*Cpr - kel_p*Cpr - k_p_t*Cpr + k_t_p*Ctr + p.k_pi*Cir
    dCir = k_rel*Cir - p.k_pi*Cir - k_i_t*Cir
    dCtr = k_rel*Ctr + k_p_t*Cpr + k_i_t*Cir - k_t_p*Ctr

    return [dCp, dCi, dCt, dCpr, dCir, dCtr]


def simulate(route='IV', modality='small_molecule', dose_mg=100.0, params: PKParams = None, t_end_h=168.0):
    p = params or PKParams()
    y0 = np.zeros(6)

    # initial conditions: put dose in plasma (IV) or peritoneal (IP), in free pool of carrier (for nanoparticle, starts in carrier pool)
    if modality == 'nanoparticle':
        if route.upper() == 'IV':
            y0 = [0.0, 0.0, 0.0, dose_mg/p.Vp, 0.0, 0.0]
        else:
            y0 = [0.0, 0.0, 0.0, 0.0, dose_mg/p.Vi, 0.0]
    else:
        if route.upper() == 'IV':
            y0 = [dose_mg/p.Vp, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            y0 = [0.0, dose_mg/p.Vi, 0.0, 0.0, 0.0, 0.0]

    sol = solve_ivp(lambda t, y: pk_ode(t, y, p, route, modality), [0, t_end_h], y0, t_eval=np.linspace(0, t_end_h, 600))
    out = pd.DataFrame(sol.y.T, columns=['Cp_free','Ci_free','Ct_free','Cp_rel','Ci_rel','Ct_rel'])
    out['t_h'] = sol.t
    out['Cp_total'] = out['Cp_free'] + out['Cp_rel']
    out['Ci_total'] = out['Ci_free'] + out['Ci_rel']
    out['Ct_total'] = out['Ct_free'] + out['Ct_rel']
    return out


def kpi(summary: pd.DataFrame):
    t = summary['t_h'].values
    auc_tumor = np.trapz(summary['Ct_total'], t)
    auc_plasma = np.trapz(summary['Cp_total'], t)
    cmax_t = summary['Ct_total'].max()
    cmax_p = summary['Cp_total'].max()
    return {
        'AUC_tumor': float(auc_tumor),
        'AUC_plasma': float(auc_plasma),
        'Cmax_tumor': float(cmax_t),
        'Cmax_plasma': float(cmax_p),
        'Tumor_to_Plasma_AUC': float(auc_tumor / (auc_plasma + 1e-9))
    }


def main():
    ap = argparse.ArgumentParser(description='Simulate PK for delivery strategies in HGSOC')
    ap.add_argument('--route', choices=['IV','IP'], default='IV')
    ap.add_argument('--modality', choices=['small_molecule','nanoparticle'], default='small_molecule')
    ap.add_argument('--dose', type=float, default=100.0)
    ap.add_argument('--k_rel', type=float, default=0.1)
    ap.add_argument('--out', type=str, default='outputs/pk_profile.csv')
    args = ap.parse_args()

    p = PKParams(k_rel=args.k_rel)
    prof = simulate(route=args.route, modality=args.modality, dose_mg=args.dose, params=p)
    m = kpi(prof)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prof.to_csv(out, index=False)
    print('KPIs:', m)
    metrics_path = out.parent / 'pk_metrics.csv'
    pd.DataFrame([m]).to_csv(metrics_path, index=False)
    print(f'Saved profile to {out} and metrics to {metrics_path}')


if __name__ == '__main__':
    main()

