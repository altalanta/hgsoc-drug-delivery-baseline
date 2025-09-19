"""Feature engineering utilities for HGSOC baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_COLUMNS = [
    "ligand_density",
    "tumor_size_cm",
    "perfusion_idx",
    "pk_clearance",
    "pk_vd",
    "pd_sensitivity",
    "exposure_auc",
    "ligand_perfusion",
    "pk_ratio",
]


@dataclass
class FeatureSpec:
    scaler: StandardScaler
    encoder: OneHotEncoder
    feature_names: list[str]


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    X_valid: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_cont_train: np.ndarray
    y_cont_valid: np.ndarray
    metadata_train: pd.DataFrame
    metadata_valid: pd.DataFrame
    spec: FeatureSpec


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    engineered["ligand_perfusion"] = engineered["ligand_density"] * engineered["perfusion_idx"]
    engineered["pk_ratio"] = engineered["pk_vd"] / np.maximum(engineered["pk_clearance"], 1e-3)
    return engineered


def transform_features(df: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    engineered = _engineer_features(df)
    numeric = engineered[NUMERIC_COLUMNS]
    numeric_scaled = spec.scaler.transform(numeric)
    stage_encoded = spec.encoder.transform(engineered[["stage"]])
    return np.hstack([numeric_scaled, stage_encoded])


def prepare_features(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> DatasetBundle:
    engineered = _engineer_features(df)
    numeric = engineered[NUMERIC_COLUMNS]
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    stage_encoded = encoder.fit_transform(engineered[["stage"]])

    feature_matrix = np.hstack([numeric_scaled, stage_encoded])
    feature_names = list(NUMERIC_COLUMNS) + list(encoder.get_feature_names_out(["stage"]))

    X_train, X_valid, y_train, y_valid, idx_train, idx_valid = train_test_split(
        feature_matrix,
        engineered["response_binary"].to_numpy(),
        np.arange(len(engineered)),
        test_size=test_size,
        random_state=seed,
        stratify=engineered["response_binary"].to_numpy(),
    )

    y_cont = engineered["response_continuous"].to_numpy()
    y_cont_train = y_cont[idx_train]
    y_cont_valid = y_cont[idx_valid]

    metadata_train = engineered.iloc[idx_train].reset_index(drop=True)
    metadata_valid = engineered.iloc[idx_valid].reset_index(drop=True)

    spec = FeatureSpec(scaler=scaler, encoder=encoder, feature_names=feature_names)

    return DatasetBundle(
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid,
        y_cont_train=y_cont_train,
        y_cont_valid=y_cont_valid,
        metadata_train=metadata_train,
        metadata_valid=metadata_valid,
        spec=spec,
    )
