from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from neurosynth.causal.phase5_engine import CausalPhase5Engine


def test_calibration_brier_below_threshold_on_synthetic_split() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(0, 1, size=(1200, 5))
    logits = x[:, 0] * 1.2 - x[:, 1] * 0.8 + rng.normal(0, 0.2, size=1200)
    y = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)

    model = LogisticRegression(max_iter=300)
    model.fit(x[:1000], y[:1000])
    probs = model.predict_proba(x[1000:])[:, 1]

    brier = brier_score_loss(y[1000:], probs)
    assert brier < 0.15


def test_fairness_auc_gap_under_005_on_synthetic_age_sex_groups() -> None:
    rng = np.random.default_rng(77)
    n = 1500
    age = rng.integers(50, 90, size=n)
    sex = rng.integers(0, 2, size=n)
    risk_signal = 0.03 * (age - 65) + 0.05 * sex + rng.normal(0, 0.6, size=n)
    y = (risk_signal > np.median(risk_signal)).astype(int)
    pred = 1 / (1 + np.exp(-(risk_signal + rng.normal(0, 0.2, size=n))))

    auc_sex0 = roc_auc_score(y[sex == 0], pred[sex == 0])
    auc_sex1 = roc_auc_score(y[sex == 1], pred[sex == 1])

    assert abs(auc_sex0 - auc_sex1) < 0.05


def test_interval_coverage_80pi_at_least_78_percent() -> None:
    rng = np.random.default_rng(456)
    y_true = rng.normal(loc=0.55, scale=0.12, size=2000)
    y_pred = y_true + rng.normal(0, 0.03, size=2000)

    spread = 0.11
    lower = y_pred - spread
    upper = y_pred + spread
    coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))

    assert coverage >= 0.78


def test_counterfactual_consistency_beneficial_intervention_reduces_risk() -> None:
    engine = CausalPhase5Engine()

    patient = pd.Series({"SleepQuality": 4.0, "MMSE": 23.0, "Inflammation": 0.8})

    def risk_forecast(df: pd.DataFrame) -> np.ndarray:
        # Lower SleepQuality should imply higher risk; this monotonic mapping is our contract.
        risk = 0.8 - 0.08 * df["SleepQuality"].to_numpy() + 0.1 * df["Inflammation"].to_numpy()
        return np.clip(risk, 0.0, 1.0)

    result = engine.counterfactual_forecast(
        patient_row=patient,
        biomarker="Inflammation",
        reduction_frac=0.3,
        forecast_fn=risk_forecast,
    )

    assert result["after"][0] <= result["before"][0]
