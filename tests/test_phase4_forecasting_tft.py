from __future__ import annotations

import numpy as np
import pandas as pd

from neurosynth.forecasting.phase4_tft import ForecastingPhase4Model, Phase4Config


def test_phase4_predict_with_uncertainty_smoke() -> None:
    model = ForecastingPhase4Model(Phase4Config(max_encoder_length=4, max_prediction_length=2, mc_samples=8))

    train = pd.DataFrame(
        {
            "patient_id": ["p1"] * 10,
            "time_idx": list(range(10)),
            "target": np.linspace(0.2, 0.8, 10),
            "age": [73.0] * 10,
            "mmse": np.linspace(28, 20, 10),
        }
    )
    val = train.copy()

    model.create_dataset(
        frame=train,
        time_idx="time_idx",
        target="target",
        group_ids=["patient_id"],
        known_reals=["time_idx", "age"],
        unknown_reals=["target", "mmse"],
    )
    model.fit(train, val)
    out = model.predict_with_uncertainty(val[["time_idx", "age", "mmse"]])

    assert "mean" in out
    assert "lower_80" in out
    assert "upper_95" in out
    assert len(out["shap_values"]) > 0
