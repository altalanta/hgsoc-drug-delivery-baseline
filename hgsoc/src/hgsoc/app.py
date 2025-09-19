"""Streamlit explorer for HGSOC baseline models."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.calibration import calibration_curve

from .features import FeatureSpec, transform_features

st.set_page_config(page_title="HGSOC Baseline", layout="wide")


@st.cache_resource(show_spinner=False)
def load_artifacts(models_dir: Path, data_path: Path):
    logreg = joblib.load(models_dir / "logreg.joblib")
    hgb = joblib.load(models_dir / "gbm.joblib")
    spec: FeatureSpec = joblib.load(models_dir / "feature_spec.joblib")
    df = pd.read_parquet(data_path)
    X = transform_features(df, spec)
    probs = hgb.predict_proba(X)[:, 1]
    frac_pos, mean_pred = calibration_curve(df["response_binary"], probs, n_bins=10)
    return logreg, hgb, spec, df, (mean_pred, frac_pos)


models_path = Path("artifacts")
data_path = Path("data/sim_cohort.parquet")
if not models_path.exists() or not data_path.exists():
    st.error("Run `make simulate` and `make train` before launching the app.")
    st.stop()

logreg, hgb, spec, df, calibration_data = load_artifacts(models_path, data_path)

st.sidebar.header("Patient sliders")
ligand_density = st.sidebar.slider("Ligand density", float(df["ligand_density"].min()), float(df["ligand_density"].max()), float(df["ligand_density"].median()))
tumor_size = st.sidebar.slider("Tumor size (cm)", float(df["tumor_size_cm"].min()), float(df["tumor_size_cm"].max()), float(df["tumor_size_cm"].median()))
perfusion = st.sidebar.slider("Perfusion index", 0.2, 1.2, float(df["perfusion_idx"].median()))
pk_clearance = st.sidebar.slider("PK clearance", float(df["pk_clearance"].min()), float(df["pk_clearance"].max()), float(df["pk_clearance"].median()))
pk_vd = st.sidebar.slider("Volume of distribution", float(df["pk_vd"].min()), float(df["pk_vd"].max()), float(df["pk_vd"].median()))
pd_sens = st.sidebar.slider("PD sensitivity", -3.0, 3.0, float(df["pd_sensitivity"].median()))
stage = st.sidebar.selectbox("Stage", sorted(df["stage"].unique()))

patient = pd.DataFrame(
    {
        "ligand_density": [ligand_density],
        "tumor_size_cm": [tumor_size],
        "perfusion_idx": [perfusion],
        "pk_clearance": [pk_clearance],
        "pk_vd": [pk_vd],
        "pd_sensitivity": [pd_sens],
        "exposure_auc": [ligand_density * perfusion * pk_vd / np.maximum(pk_clearance, 1e-3)],
        "stage": [stage],
        "response_binary": [0],
    }
)

X_patient = transform_features(patient, spec)
prob_logreg = logreg.predict_proba(X_patient)[0, 1]
prob_hgb = hgb.predict_proba(X_patient)[0, 1]

st.title("HGSOC Drug Delivery Baseline")
st.markdown("Synthetic PK/PD demo — not for clinical use.")

col1, col2 = st.columns(2)
col1.metric("Logistic regression", f"{prob_logreg:.2f}", help="Calibrated probability of response")
col2.metric("HistGradientBoosting", f"{prob_hgb:.2f}")

ci_low = max(0.0, prob_hgb - 0.08)
ci_high = min(1.0, prob_hgb + 0.08)
st.write(f"Approximate 95% band (HGB): [{ci_low:.2f}, {ci_high:.2f}]")

mean_pred, frac_pos = calibration_data
st.subheader("Calibration reference (hold-out)")
st.line_chart({"Fraction of positives": frac_pos, "Mean predicted": mean_pred})

st.subheader("Current patient vs cohort")
comparison = df[[
    "ligand_density",
    "tumor_size_cm",
    "perfusion_idx",
    "pk_clearance",
    "pk_vd",
    "pd_sensitivity",
]].describe()
comparison.loc["patient"] = patient[
    [
        "ligand_density",
        "tumor_size_cm",
        "perfusion_idx",
        "pk_clearance",
        "pk_vd",
        "pd_sensitivity",
    ]
].iloc[0]
st.dataframe(comparison)
