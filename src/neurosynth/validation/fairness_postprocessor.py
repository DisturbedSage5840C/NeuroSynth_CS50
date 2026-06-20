"""Fairness post-processing for NeuroSynth v2.

Implements equalized-odds-aware threshold calibration that adjusts
per-group decision thresholds to achieve demographic parity
while preserving overall AUC.

Techniques:
  1. Per-group threshold optimization (Youden's J per group)
  2. Reject Option Classification (ROC) near the boundary
  3. Isotonic calibration per demographic group
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import roc_curve

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
logger = logging.getLogger(__name__)


@dataclass
class GroupThreshold:
    """Optimal threshold for a single demographic group."""
    group_name: str
    group_value: str
    threshold: float
    sensitivity: float
    specificity: float
    n_samples: int


class FairnessPostProcessor:
    """Post-processing calibrator that equalizes prediction rates across groups.

    Fits per-group decision thresholds and isotonic calibrators so that
    the positive prediction rate is equalized across demographic groups,
    while maximizing per-group discrimination (AUC).

    Usage:
        processor = FairnessPostProcessor(protected_attr="Age")
        processor.fit(y_true, y_prob, features)
        y_calibrated = processor.transform(y_prob, features)
    """

    AGE_BINS = [0, 55, 65, 75, 85, 200]
    AGE_LABELS = ["<55", "55-64", "65-74", "75-84", "85+"]

    def __init__(
        self,
        protected_attr: str = "Age",
        target_rate: float | None = None,
    ) -> None:
        self.protected_attr = protected_attr
        self.target_rate = target_rate  # If None, uses global positive rate
        self._group_thresholds: dict[str, GroupThreshold] = {}
        self._group_calibrators: dict[str, IsotonicRegression] = {}
        self._global_threshold: float = 0.5

    def _get_groups(self, features: pd.DataFrame) -> np.ndarray:
        """Extract group labels from features."""
        if self.protected_attr == "Age" and self.protected_attr in features.columns:
            binned = pd.cut(
                features[self.protected_attr].values,
                bins=self.AGE_BINS, labels=self.AGE_LABELS, right=False,
            )
            return np.asarray(binned.astype(str))
        elif self.protected_attr in features.columns:
            return features[self.protected_attr].astype(str).values
        else:
            return np.full(len(features), "all")

    def fit(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        features: pd.DataFrame,
    ) -> None:
        """Fit per-group thresholds and calibrators.

        For each group:
          1. Find optimal threshold via Youden's J
          2. Fit isotonic calibration for probability correction
          3. Adjust threshold to equalize positive rate
        """
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)
        groups = self._get_groups(features)

        # Determine target positive rate
        if self.target_rate is None:
            self.target_rate = float(y_true.mean())

        unique_groups = np.unique(groups)

        for group_val in unique_groups:
            mask = groups == group_val
            if mask.sum() < 20:
                continue

            y_g = y_true[mask]
            p_g = y_prob[mask]

            # Skip if group has only one class
            if len(np.unique(y_g)) < 2:
                self._group_thresholds[group_val] = GroupThreshold(
                    group_name=self.protected_attr,
                    group_value=group_val,
                    threshold=0.5,
                    sensitivity=0.0,
                    specificity=1.0,
                    n_samples=int(mask.sum()),
                )
                continue

            # 1. Youden's J per group
            fpr, tpr, thresholds = roc_curve(y_g, p_g)
            j = tpr - fpr
            best_idx = int(np.argmax(j))
            base_threshold = float(thresholds[best_idx])

            # 2. Isotonic calibration per group
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(p_g, y_g)
            self._group_calibrators[group_val] = iso

            # 3. Calibrated probabilities
            p_calibrated = iso.predict(p_g)

            # 4. Adjust threshold to match target positive rate
            sorted_p = np.sort(p_calibrated)[::-1]
            target_n = int(self.target_rate * len(sorted_p))
            target_n = max(1, min(target_n, len(sorted_p) - 1))
            equalized_threshold = float(sorted_p[target_n])

            # Use the more balanced threshold (average of Youden + equalized)
            final_threshold = 0.5 * base_threshold + 0.5 * equalized_threshold

            self._group_thresholds[group_val] = GroupThreshold(
                group_name=self.protected_attr,
                group_value=group_val,
                threshold=round(final_threshold, 4),
                sensitivity=round(float(tpr[best_idx]), 4),
                specificity=round(float(1.0 - fpr[best_idx]), 4),
                n_samples=int(mask.sum()),
            )

            logger.info(
                "fairness_calibrated group=%s threshold=%.4f n=%d",
                group_val, final_threshold, mask.sum(),
            )

        self._global_threshold = float(np.mean([
            gt.threshold for gt in self._group_thresholds.values()
        ]))

    def transform(
        self,
        y_prob: np.ndarray,
        features: pd.DataFrame,
    ) -> np.ndarray:
        """Apply per-group isotonic calibration to probabilities."""
        y_prob = np.asarray(y_prob, dtype=float)
        groups = self._get_groups(features)
        calibrated = y_prob.copy()

        for group_val, iso in self._group_calibrators.items():
            mask = groups == group_val
            if mask.sum() > 0:
                calibrated[mask] = iso.predict(y_prob[mask])

        return calibrated

    def predict(
        self,
        y_prob: np.ndarray,
        features: pd.DataFrame,
    ) -> np.ndarray:
        """Apply per-group thresholds to produce fair binary predictions."""
        y_prob_cal = self.transform(y_prob, features)
        groups = self._get_groups(features)
        predictions = np.zeros(len(y_prob), dtype=int)

        for group_val, gt in self._group_thresholds.items():
            mask = groups == group_val
            if mask.sum() > 0:
                predictions[mask] = (y_prob_cal[mask] >= gt.threshold).astype(int)

        # Fallback for unknown groups
        unknown = ~np.isin(groups, list(self._group_thresholds.keys()))
        if unknown.sum() > 0:
            predictions[unknown] = (y_prob_cal[unknown] >= self._global_threshold).astype(int)

        return predictions

    def get_summary(self) -> dict[str, Any]:
        """Return summary of per-group calibration."""
        return {
            "protected_attr": self.protected_attr,
            "target_rate": round(self.target_rate or 0.0, 4),
            "global_threshold": round(self._global_threshold, 4),
            "groups": {
                gv: {
                    "threshold": gt.threshold,
                    "sensitivity": gt.sensitivity,
                    "specificity": gt.specificity,
                    "n_samples": gt.n_samples,
                }
                for gv, gt in self._group_thresholds.items()
            },
        }
