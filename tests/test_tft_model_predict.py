from __future__ import annotations

import numpy as np
import pandas as pd

from neurosynth.temporal_tft.model import NeuroTFT


class DummyModel:
    def predict(self, patient_df, mode="raw", return_x=False):
        _ = patient_df
        pred = np.array([[[40, 42, 44, 45, 46, 47, 48], [42, 44, 46, 47, 48, 49, 50], [44, 46, 48, 49, 50, 51, 52], [46, 48, 50, 51, 52, 53, 54], [48, 50, 52, 53, 54, 55, 56], [50, 52, 54, 55, 56, 57, 58]]], dtype=np.float32)

        class Wrap:
            def __init__(self, arr):
                self._arr = arr

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._arr

        raw = {"prediction": Wrap(pred), "attention": Wrap(np.ones((1, 8, 6), dtype=np.float32))}
        if return_x:
            return raw, {}
        return raw

    def interpret_output(self, raw, reduction="mean"):
        _ = (raw, reduction)
        return {"encoder_variables": {"nfl_plasma": np.array([0.3, 0.4]), "delta_hippocampus": np.array([0.5, 0.6])}}


def test_predict_with_uncertainty_monotonic() -> None:
    m = NeuroTFT(DummyModel())
    out = m.predict_with_uncertainty(pd.DataFrame({"x": [1]}))
    med = out["median_forecast"]
    assert np.all(med[1:] >= med[:-1])
    assert out["prediction_interval_80"].shape == (6, 2)
