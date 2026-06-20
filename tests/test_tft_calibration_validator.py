from __future__ import annotations

import numpy as np

from neurosynth.temporal_tft.calibration import TFTCalibrator, TFTValidator


class DummyTensor:
    def __init__(self, arr):
        self.arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class DummyModel:
    def predict(self, loader, mode="raw"):
        _ = (loader, mode)
        pred = np.random.RandomState(42).rand(12, 6, 7).astype(np.float32) * 80
        target = np.random.RandomState(7).rand(12, 6, 2).astype(np.float32) * 80
        return {"prediction": DummyTensor(pred), "target_scale": DummyTensor(target)}


def test_calibrator_and_validator_outputs() -> None:
    model = DummyModel()
    calibrator = TFTCalibrator()
    calibrated = calibrator.calibrate_quantiles(model, val_loader=object())

    assert not calibrated.coverage_table.empty
    validator = TFTValidator()
    report = validator.comprehensive_validation(model, test_loader=object())

    assert "rmse" in report.overall_metrics
    assert "coverage80" in report.calibration_metrics
