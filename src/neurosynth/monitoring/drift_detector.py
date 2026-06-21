# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""Drift Detector — PSI & KS-based feature drift detection.

Tiered thresholds:
  PSI < 0.10        → ✅ NO_DRIFT
  PSI 0.10 - 0.20   → ⚠️ MINOR — log only
  PSI 0.20 - 0.25   → 🟡 WARNING — alert, increase monitoring
  PSI ≥ 0.25        → 🔴 CRITICAL — trigger auto-retrain
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DriftSeverity(str, Enum):
    NO_DRIFT = "NO_DRIFT"
    MINOR = "MINOR"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class FeatureDrift:
    """Drift result for a single feature."""
    feature: str
    psi: float
    ks_stat: float
    ks_pvalue: float
    severity: DriftSeverity
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float


@dataclass
class DriftReport:
    """Aggregated drift report across all features."""
    timestamp: str
    total_features: int
    drifted_features: int
    feature_results: list[FeatureDrift] = field(default_factory=list)
    overall_severity: DriftSeverity = DriftSeverity.NO_DRIFT
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_features": self.total_features,
            "drifted_features": self.drifted_features,
            "overall_severity": self.overall_severity.value,
            "recommendation": self.recommendation,
            "features": [
                {
                    "feature": f.feature,
                    "psi": round(f.psi, 6),
                    "ks_stat": round(f.ks_stat, 6),
                    "ks_pvalue": round(f.ks_pvalue, 6),
                    "severity": f.severity.value,
                    "ref_mean": round(f.reference_mean, 4),
                    "cur_mean": round(f.current_mean, 4),
                }
                for f in self.feature_results
            ],
        }


# PSI thresholds
PSI_MINOR = 0.10
PSI_WARNING = 0.20
PSI_CRITICAL = 0.25


def _classify_severity(psi: float) -> DriftSeverity:
    if psi >= PSI_CRITICAL:
        return DriftSeverity.CRITICAL
    elif psi >= PSI_WARNING:
        return DriftSeverity.WARNING
    elif psi >= PSI_MINOR:
        return DriftSeverity.MINOR
    return DriftSeverity.NO_DRIFT


def _compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Compute Population Stability Index between reference and current distributions."""
    eps = 1e-8
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    bins = np.linspace(min_val - eps, max_val + eps, n_bins + 1)

    ref_hist, _ = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bins)

    ref_pct = ref_hist / max(ref_hist.sum(), 1) + eps
    cur_pct = cur_hist / max(cur_hist.sum(), 1) + eps

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return max(psi, 0.0)


def _compute_ks(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Compute two-sample Kolmogorov-Smirnov statistic."""
    try:
        from scipy.stats import ks_2samp
        stat, pvalue = ks_2samp(reference, current)
        return float(stat), float(pvalue)
    except ImportError:
        # Fallback: manual KS
        all_values = np.concatenate([reference, current])
        all_values.sort()
        n1, n2 = len(reference), len(current)
        cdf1 = np.searchsorted(np.sort(reference), all_values, side="right") / n1
        cdf2 = np.searchsorted(np.sort(current), all_values, side="right") / n2
        stat = float(np.max(np.abs(cdf1 - cdf2)))
        return stat, 0.0  # p-value requires scipy


class DriftDetector:
    """Production drift detector with PSI + KS tests and tiered alerting.

    Usage:
        detector = DriftDetector()
        report = detector.detect(reference_df, current_df, feature_names)
        if report.overall_severity == DriftSeverity.CRITICAL:
            trigger_retrain()
    """

    def __init__(
        self,
        psi_bins: int = 10,
        ks_alpha: float = 0.05,
        min_samples: int = 30,
    ) -> None:
        self.psi_bins = psi_bins
        self.ks_alpha = ks_alpha
        self.min_samples = min_samples

    def detect(
        self,
        reference: np.ndarray | Any,
        current: np.ndarray | Any,
        feature_names: list[str] | None = None,
    ) -> DriftReport:
        """Run drift detection across all features.

        Args:
            reference: Reference dataset (n_samples, n_features) or DataFrame
            current: Current dataset (same shape)
            feature_names: Column names (auto-generated if None)
        """
        ref = np.asarray(reference)
        cur = np.asarray(current)

        if ref.ndim == 1:
            ref = ref.reshape(-1, 1)
            cur = cur.reshape(-1, 1)

        n_features = ref.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        results: list[FeatureDrift] = []
        worst = DriftSeverity.NO_DRIFT

        for i, fname in enumerate(feature_names[:n_features]):
            ref_col = ref[:, i][~np.isnan(ref[:, i])]
            cur_col = cur[:, i][~np.isnan(cur[:, i])]

            if len(ref_col) < self.min_samples or len(cur_col) < self.min_samples:
                logger.warning("drift_skip feature=%s ref_n=%d cur_n=%d (below min_samples)", fname, len(ref_col), len(cur_col))
                continue

            psi = _compute_psi(ref_col, cur_col, self.psi_bins)
            ks_stat, ks_pvalue = _compute_ks(ref_col, cur_col)
            severity = _classify_severity(psi)

            if severity.value > worst.value or (
                severity == DriftSeverity.CRITICAL and worst != DriftSeverity.CRITICAL
            ):
                worst = severity

            results.append(FeatureDrift(
                feature=fname,
                psi=psi,
                ks_stat=ks_stat,
                ks_pvalue=ks_pvalue,
                severity=severity,
                reference_mean=float(np.mean(ref_col)),
                current_mean=float(np.mean(cur_col)),
                reference_std=float(np.std(ref_col)),
                current_std=float(np.std(cur_col)),
            ))

        # Overall severity = worst individual
        sev_order = [DriftSeverity.NO_DRIFT, DriftSeverity.MINOR, DriftSeverity.WARNING, DriftSeverity.CRITICAL]
        overall = DriftSeverity.NO_DRIFT
        for r in results:
            if sev_order.index(r.severity) > sev_order.index(overall):
                overall = r.severity

        drifted_count = sum(1 for r in results if r.severity != DriftSeverity.NO_DRIFT)

        recommendations = {
            DriftSeverity.NO_DRIFT: "All features stable. No action required.",
            DriftSeverity.MINOR: f"{drifted_count} features show minor drift. Continue monitoring.",
            DriftSeverity.WARNING: f"{drifted_count} features drifting. Increase monitoring frequency and review data pipeline.",
            DriftSeverity.CRITICAL: f"{drifted_count} features critically drifted. Trigger model retraining immediately.",
        }

        report = DriftReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_features=len(results),
            drifted_features=drifted_count,
            feature_results=sorted(results, key=lambda x: x.psi, reverse=True),
            overall_severity=overall,
            recommendation=recommendations[overall],
        )

        logger.info(
            "drift_detection_complete features=%d drifted=%d severity=%s",
            report.total_features, report.drifted_features, report.overall_severity.value,
        )
        return report

    # ------------------------------------------------------------------
    # Auto-retrain trigger (wires drift -> training pipeline)
    # ------------------------------------------------------------------

    def trigger_retrain(
        self,
        report: DriftReport,
        *,
        retrain_window_days: int = 90,
        dispatcher: Any | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Dispatch a retraining job in response to drift.

        Decoupled from the backend: dispatches the Celery task ``run_full_training_pipeline``
        *by name* over the broker (NEUROSYNTH_REDIS_URL) so this module never imports
        ``backend``. If Celery / the broker is unavailable (or ``dry_run``), the trigger
        is logged and returned without raising — drift detection must never crash on a
        dispatch failure. Pass ``dispatcher`` (a callable taking the payload, returning a
        task id) to override the transport, e.g. in tests.
        """
        import os

        payload = {
            "trigger_reason": f"drift_{report.overall_severity.value.lower()}",
            "severity": report.overall_severity.value,
            "drift_features": [
                f.feature for f in report.feature_results
                if f.severity == DriftSeverity.CRITICAL
            ],
            "psi_max": round(max((f.psi for f in report.feature_results), default=0.0), 6),
            "retrain_window_days": retrain_window_days,
        }

        if dispatcher is not None:
            task_id = dispatcher(payload)
            logger.info("auto_retrain_triggered via=custom task_id=%s severity=%s", task_id, payload["severity"])
            return {"status": "dispatched", "task_id": task_id, **payload}

        if dry_run:
            logger.info("auto_retrain_logged (dry_run) severity=%s features=%s",
                        payload["severity"], payload["drift_features"])
            return {"status": "logged", **payload}

        try:
            from celery import Celery

            broker = os.getenv("NEUROSYNTH_REDIS_URL", "redis://localhost:6379/0")
            app = Celery("neurosynth-drift", broker=broker, backend=broker)
            res = app.send_task("run_full_training_pipeline", kwargs=payload, queue="training")
            logger.info("auto_retrain_triggered task_id=%s severity=%s", res.id, payload["severity"])
            return {"status": "dispatched", "task_id": res.id, **payload}
        except Exception as exc:  # broker down / celery missing -> log, never crash
            logger.warning("auto_retrain_dispatch_failed error=%s (logged only)", exc)
            return {"status": "logged", "error": str(exc), **payload}

    def detect_and_maybe_retrain(
        self,
        reference: np.ndarray | Any,
        current: np.ndarray | Any,
        feature_names: list[str] | None = None,
        *,
        dispatcher: Any | None = None,
        dry_run: bool = False,
    ) -> tuple[DriftReport, dict[str, Any] | None]:
        """Run detection and auto-trigger retraining when severity is CRITICAL."""
        report = self.detect(reference, current, feature_names)
        action = None
        if report.overall_severity == DriftSeverity.CRITICAL:
            action = self.trigger_retrain(report, dispatcher=dispatcher, dry_run=dry_run)
        return report, action
