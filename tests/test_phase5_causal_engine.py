from __future__ import annotations

import numpy as np
import pandas as pd

from neurosynth.causal.phase5_engine import CausalPhase5Engine, Phase5Config


def test_phase5_counterfactual_and_uncertainty_smoke() -> None:
    engine = CausalPhase5Engine(Phase5Config(bootstrap_samples=10))

    frame = pd.DataFrame(
        {
            "SleepQuality": np.random.uniform(3, 8, size=32),
            "MMSE": np.random.uniform(18, 30, size=32),
            "DiagnosisRisk": np.random.uniform(0, 1, size=32),
        }
    )

    out = engine.predict_with_uncertainty(frame[["SleepQuality", "MMSE"]], frame["DiagnosisRisk"])
    assert "mean" in out
    assert "lower_80" in out
    assert "shap_values" in out

    patient = frame.iloc[0]

    def fake_forecast(df: pd.DataFrame) -> np.ndarray:
        return df[["SleepQuality", "MMSE"]].mean(axis=1).to_numpy()

    cf = engine.counterfactual_forecast(patient, biomarker="SleepQuality", reduction_frac=0.2, forecast_fn=fake_forecast)
    assert "before" in cf and "after" in cf and "delta" in cf
