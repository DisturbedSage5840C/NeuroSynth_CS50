from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report
from prometheus_client import Gauge, Histogram

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

@dataclass
class DriftReport:
    feature_scores: dict[str, float]
    drifted_features: list[str]


@dataclass
class CalibrationMetrics:
    mae: float
    interval_coverage: float


class ModelDriftDetector:
    def detect_data_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame, feature_names: list[str]) -> DriftReport:
        rep = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        rep.run(reference_data=reference_data[feature_names], current_data=current_data[feature_names])

        scores = {}
        drifted = []
        for f in feature_names:
            ref = reference_data[f].dropna().to_numpy()
            cur = current_data[f].dropna().to_numpy()
            if len(ref) == 0 or len(cur) == 0:
                scores[f] = 0.0
                continue
            stat = float(np.abs(np.nanmean(ref) - np.nanmean(cur)) / (np.nanstd(ref) + 1e-6))
            scores[f] = stat
            if stat > 0.15:
                drifted.append(f)
        _ = rep
        return DriftReport(feature_scores=scores, drifted_features=drifted)

    def detect_prediction_drift(self, reference_preds: np.ndarray, current_preds: np.ndarray) -> bool:
        bins = np.linspace(min(reference_preds.min(), current_preds.min()), max(reference_preds.max(), current_preds.max()), 11)
        ref_hist, _ = np.histogram(reference_preds, bins=bins)
        cur_hist, _ = np.histogram(current_preds, bins=bins)
        ref_pct = ref_hist / max(ref_hist.sum(), 1)
        cur_pct = cur_hist / max(cur_hist.sum(), 1)
        psi = np.sum((cur_pct - ref_pct) * np.log((cur_pct + 1e-8) / (ref_pct + 1e-8)))
        return bool(psi > 0.25)


class ClinicalOutcomeMonitor:
    def track_prediction_calibration(self, predicted_dcis: np.ndarray, actual_dcis: np.ndarray, horizon_months: int) -> CalibrationMetrics:
        mae = float(np.mean(np.abs(predicted_dcis - actual_dcis)))
        lower = predicted_dcis - 5.0
        upper = predicted_dcis + 5.0
        cov = float(np.mean((actual_dcis >= lower) & (actual_dcis <= upper)))
        _ = horizon_months
        return CalibrationMetrics(mae=mae, interval_coverage=cov)


class NeuroSynthMonitor:
    def __init__(self) -> None:
        self.latency = Histogram("neurosynth_inference_latency_seconds", "Inference latency", ["model", "modality"])
        self.requests = Histogram("neurosynth_requests_total", "Requests", ["endpoint", "status"])
        self.drift = Gauge("neurosynth_data_drift_score", "Data drift score", ["feature"])
        self.mae = Gauge("neurosynth_prediction_mae", "Prediction MAE", ["horizon_months", "disease"])
        self.gpu_mem = Gauge("neurosynth_gpu_memory_used_bytes", "GPU memory", ["pod", "gpu_index"])
        self.pipeline_duration = Histogram("neurosynth_pipeline_duration_seconds", "Pipeline duration", ["stage"])

    def update_drift_metrics(self, report: DriftReport) -> None:
        for f, s in report.feature_scores.items():
            self.drift.labels(feature=f).set(float(s))

    def grafana_dashboard_json(self) -> dict:
        return {
            "title": "NeuroSynth Production",
            "panels": [
                {"title": "Request Rate & Error Rate", "type": "timeseries"},
                {"title": "GPU Utilization per Model", "type": "timeseries"},
                {"title": "Pipeline Latency P50/P95/P99", "type": "timeseries"},
                {"title": "Data Drift Scores", "type": "timeseries"},
                {"title": "Prediction Calibration", "type": "timeseries"},
                {"title": "Active Patient Analyses", "type": "stat"},
            ],
        }
