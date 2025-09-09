HGSOC Drug Delivery Baseline

Overview
- Goal: Demonstrate end-to-end reasoning for drug delivery in High-Grade Serous Ovarian Cancer (HGSOC) using synthetic data: target prioritization from tumor expression, formulation/route choices, and simple PK/PD delivery simulations to compare strategies.
- Scope: Synthetic cohort generation, target scoring, delivery modality choice (IV/IP; small molecule, ADC, ligand-targeted nanoparticle), basic three-compartment PK for tumor exposure, and clear plots/CSV outputs.
- Note: All data are synthetic and for demonstration only.

Project Structure
- `data/annotations/surface_targets.csv` — Minimal target metadata (e.g., FOLR1, MSLN, MUC16, EGFR, CLDN6).
- `scripts/generate_synthetic_hgsoc_cohort.py` — Creates a synthetic patient cohort with target expression, HRD/platinum status, tumor burden, ascites, etc.
- `analysis/target_prioritization.py` — Combines cohort expression with annotations to rank targets per-patient and at population level.
- `modeling/pkpd_delivery_sim.py` — ODE-based delivery model for IV/IP routes and modalities (small molecule vs nanoparticle-like), computes AUC/Cmax metrics.
- `analysis/design_and_selection.py` — Orchestrates: pick a target and route, sweep formulation parameters, run simulations, produce candidate rankings and figures.
- `outputs/` — Saved CSVs and figures.

Quick Start
1) Install dependencies
   - `pip install -r requirements.txt`
2) Generate a synthetic cohort (100 patients)
   - `python scripts/generate_synthetic_hgsoc_cohort.py --n 100 --out data/synthetic_hgsoc_cohort.csv`
3) Rank targets
   - `python analysis/target_prioritization.py --cohort data/synthetic_hgsoc_cohort.csv --annotations data/annotations/surface_targets.csv --outdir outputs`
4) Design & simulate delivery strategies
   - `python analysis/design_and_selection.py --cohort data/synthetic_hgsoc_cohort.csv --annotations data/annotations/surface_targets.csv --target FOLR1 --route IP --outdir outputs`

What It Demonstrates
- Domain-aware feature engineering: target surface accessibility, internalization, tumor-to-normal expression, and delivery route feasibility (e.g., IP in HGSOC with ascites).
- Model-driven comparison: contrasts IV vs IP and small molecules vs nanoparticles for tumor AUC vs systemic exposure.
- Decision support artifacts: CSV rankings and plots for communication.

Assumptions & Limitations
- Simplified PK: three compartments (plasma, peritoneal, tumor interstitium) with first-order transfers; parameters are illustrative.
- Target biology is represented by coarse annotations; replace with real consortia data (e.g., CPTAC/TCGA) when available.
- No clinical recommendations; for educational/portfolio purposes only.

Next Extensions
- Calibrate PK/PD parameters to literature per modality and particle size; add receptor binding/internalization kinetics (TMDD) for ADCs.
- Add multi-objective optimization (e.g., NSGA-II) over particle size, release half-life, and dose.
- Integrate public expression datasets (after curation) to move beyond synthetic data.

License
- MIT (see LICENSE)
