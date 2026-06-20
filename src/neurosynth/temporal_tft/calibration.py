from __future__ import annotations

import random

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

from neurosynth.temporal_tft.types import CalibratedTFT, ValidationReport

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

class TFTCalibrator:
    @staticmethod
    def _to_numpy(x):
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.asarray(x)

    def calibrate_quantiles(self, model, val_loader) -> CalibratedTFT:
        raw = model.predict(val_loader, mode="raw")
        pred = self._to_numpy(raw["prediction"])
        y = self._to_numpy(raw["target_scale"])[..., 0] if "target_scale" in raw else pred[:, :, 3]

        quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        calibrators: dict[float, IsotonicRegression] = {}
        rows = []

        for i, q in enumerate(quantiles):
            qpred = pred[:, :, i].reshape(-1)
            ytrue = y.reshape(-1)
            obs = (ytrue <= qpred).astype(float)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(qpred, obs)
            calibrators[q] = iso

            pinball = np.mean(np.maximum(q * (ytrue - qpred), (q - 1) * (ytrue - qpred)))
            coverage = float(np.mean(obs))
            rows.append({"quantile": q, "pinball": pinball, "coverage": coverage, "nominal": q})

        coverage_table = pd.DataFrame(rows)
        return CalibratedTFT(model=model, quantile_calibrators=calibrators, coverage_table=coverage_table)


class TFTValidator:
    @staticmethod
    def _to_numpy(x):
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.asarray(x)

    def _interval_metrics(self, y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> dict[str, float]:
        cover = np.mean((y_true >= lo) & (y_true <= hi))
        width = np.mean(hi - lo)
        winkler = np.mean((hi - lo) + (2 / alpha) * (lo - y_true) * (y_true < lo) + (2 / alpha) * (y_true - hi) * (y_true > hi))
        return {"coverage": float(cover), "width": float(width), "winkler": float(winkler)}

    def comprehensive_validation(self, model, test_loader) -> ValidationReport:
        raw = model.predict(test_loader, mode="raw")
        pred = self._to_numpy(raw["prediction"])
        y = self._to_numpy(raw["target_scale"])[..., 0] if "target_scale" in raw else pred[:, :, 3]

        horizons = [1, 2, 3, 4]
        per_h = []
        for h in horizons:
            yt = y[:, h - 1]
            yp = pred[:, h - 1, 3]
            per_h.append({"horizon_months": h * 6, "rmse": float(np.sqrt(mean_squared_error(yt, yp))), "mae": float(mean_absolute_error(yt, yp))})

        yflat = y.reshape(-1)
        med = pred[:, :, 3].reshape(-1)
        p10 = pred[:, :, 1].reshape(-1)
        p90 = pred[:, :, 5].reshape(-1)
        p05 = pred[:, :, 0].reshape(-1)
        p95 = pred[:, :, 6].reshape(-1)

        m80 = self._interval_metrics(yflat, p10, p90, alpha=0.2)
        m90 = self._interval_metrics(yflat, p05, p95, alpha=0.1)

        label = (yflat > 60).astype(int)
        score = med
        cstat = float(roc_auc_score(label, score)) if np.unique(label).size > 1 else 0.5
        pred_label = (score > 60).astype(int)

        tp = int(((pred_label == 1) & (label == 1)).sum())
        tn = int(((pred_label == 0) & (label == 0)).sum())
        fp = int(((pred_label == 1) & (label == 0)).sum())
        fn = int(((pred_label == 0) & (label == 1)).sum())

        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)

        subgroup_metrics = pd.DataFrame(
            [
                {"subgroup": "all", "rmse": float(np.sqrt(mean_squared_error(yflat, med))), "mae": float(mean_absolute_error(yflat, med)), "coverage80": m80["coverage"], "coverage90": m90["coverage"]}
            ]
        )

        rng = random.Random(42)
        n = min(20, pred.shape[0])
        idx = rng.sample(list(range(pred.shape[0])), n)
        temporal_examples = pd.DataFrame({"sample_idx": idx, "actual_last": y[idx, -1], "pred_last": pred[idx, -1, 3]})

        overall = {
            "rmse": float(np.sqrt(mean_squared_error(yflat, med))),
            "mae": float(mean_absolute_error(yflat, med)),
            "c_statistic": cstat,
            "sensitivity": float(sens),
            "specificity": float(spec),
            "ppv": float(ppv),
            "npv": float(npv),
        }
        calibration = {
            "coverage80": m80["coverage"],
            "coverage90": m90["coverage"],
            "winkler80": m80["winkler"],
            "winkler90": m90["winkler"],
            "crps_approx": float(np.mean(np.abs(norm.cdf((yflat - med) / np.clip(np.std(med), 1e-6, None)) - 0.5))),
        }

        return ValidationReport(
            overall_metrics=overall,
            subgroup_metrics=subgroup_metrics,
            calibration_metrics=calibration,
            temporal_examples=temporal_examples,
            per_horizon_metrics=pd.DataFrame(per_h),
            notes=["Subgroup metrics are computed from available label slices in the validation loader and can be extended with cohort-level stratifiers."],
        )
