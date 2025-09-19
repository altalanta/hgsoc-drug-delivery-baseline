import numpy as np
from sklearn.linear_model import LogisticRegression

from hgsoc.features import prepare_features
from hgsoc.simulate import simulate_cohort
from hgsoc.eval import _calibration_params


def test_logistic_brier_and_calibration() -> None:
    df = simulate_cohort(200, seed=321)
    bundle = prepare_features(df, test_size=0.3, seed=99)
    model = LogisticRegression(max_iter=500, class_weight="balanced")
    model.fit(bundle.X_train, bundle.y_train)
    probs = model.predict_proba(bundle.X_valid)[:, 1]
    brier = np.mean((probs - bundle.y_valid) ** 2)
    assert brier < 0.25
    calib = _calibration_params(bundle.y_valid, probs)
    assert 0.8 < calib["slope"] < 1.2
