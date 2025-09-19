from pathlib import Path

import numpy as np

from hgsoc.simulate import simulate_cohort


def test_simulate_schema_and_determinism(tmp_path: Path) -> None:
    df_a = simulate_cohort(100, seed=123)
    df_b = simulate_cohort(100, seed=123)
    assert list(df_a.columns) == [
        "ligand_density",
        "tumor_size_cm",
        "perfusion_idx",
        "pk_clearance",
        "pk_vd",
        "pd_sensitivity",
        "exposure_auc",
        "stage",
        "response_binary",
        "response_continuous",
    ]
    assert df_a.equals(df_b)
    assert 0 < df_a["response_binary"].mean() < 1
    assert df_a["response_continuous"].between(0, 100).all()
