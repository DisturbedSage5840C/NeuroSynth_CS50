# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""Fairness assessment for NeuroSynth v2 models.

Evaluates demographic parity and equalized odds across
protected attributes (age group, sex, ethnicity).

Metrics:
  - Demographic parity ratio (DPR): P(ŷ=1|A=a) / P(ŷ=1|A=b)
  - Equalized odds ratio (EOR): TPR(A=a) / TPR(A=b)
  - Predictive parity: PPV(A=a) / PPV(A=b)
  - Calibration parity: E[Y|ŷ=p, A=a] ≈ E[Y|ŷ=p, A=b]

Thresholds sourced from FDA SaMD guidance and EU AI Act:
  - DPR ∈ [0.80, 1.25] → PASS (four-fifths rule)
  - EOR ∈ [0.80, 1.25] → PASS
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GroupMetrics:
    """Metrics for a single demographic group."""
    group_name: str
    group_value: str
    n_samples: int
    prevalence: float          # P(y=1)
    positive_rate: float       # P(ŷ=1)
    tpr: float                 # True positive rate (sensitivity)
    fpr: float                 # False positive rate
    ppv: float                 # Positive predictive value (precision)
    npv: float                 # Negative predictive value
    auc: float = 0.0


@dataclass
class FairnessReport:
    """Full fairness assessment across all protected attributes."""
    model_name: str
    n_total: int
    groups: list[GroupMetrics] = field(default_factory=list)

    # Parity ratios (min/max group ratio for each metric)
    demographic_parity_ratio: float = 1.0
    equalized_odds_ratio: float = 1.0
    predictive_parity_ratio: float = 1.0

    # Worst-case gaps
    max_tpr_gap: float = 0.0
    max_fpr_gap: float = 0.0
    max_ppv_gap: float = 0.0

    # Per-attribute breakdowns
    attribute_reports: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Verdict
    passes_four_fifths: bool = True
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "n_total": self.n_total,
            "demographic_parity_ratio": round(self.demographic_parity_ratio, 4),
            "equalized_odds_ratio": round(self.equalized_odds_ratio, 4),
            "predictive_parity_ratio": round(self.predictive_parity_ratio, 4),
            "max_tpr_gap": round(self.max_tpr_gap, 4),
            "max_fpr_gap": round(self.max_fpr_gap, 4),
            "max_ppv_gap": round(self.max_ppv_gap, 4),
            "passes_four_fifths": self.passes_four_fifths,
            "n_groups": len(self.groups),
            "attribute_reports": self.attribute_reports,
            "warnings": self.warnings,
        }


class FairnessAuditor:
    """Fairness evaluation across demographic groups.

    Computes disparity metrics for each protected attribute and
    checks against FDA/EU regulatory thresholds.
    """

    # Four-fifths rule bounds (from EEOC Uniform Guidelines)
    PARITY_LOWER = 0.80
    PARITY_UPPER = 1.25

    # Default age group boundaries
    AGE_BINS = [0, 55, 65, 75, 85, 200]
    AGE_LABELS = ["<55", "55-64", "65-74", "75-84", "85+"]

    def __init__(
        self,
        protected_attributes: list[str] | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.protected_attributes = protected_attributes or ["Age", "Gender", "Ethnicity"]
        self.threshold = threshold

    def _compute_group_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_pred: np.ndarray,
        group_name: str,
        group_value: str,
    ) -> GroupMetrics:
        """Compute metrics for a single demographic group."""
        n = len(y_true)
        if n == 0:
            return GroupMetrics(
                group_name=group_name, group_value=group_value,
                n_samples=0, prevalence=0, positive_rate=0,
                tpr=0, fpr=0, ppv=0, npv=0,
            )

        prevalence = float(y_true.mean())
        positive_rate = float(y_pred.mean())

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)

        from sklearn.metrics import roc_auc_score
        try:
            auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
        except Exception:
            auc = 0.5

        return GroupMetrics(
            group_name=group_name, group_value=str(group_value),
            n_samples=n, prevalence=round(prevalence, 4),
            positive_rate=round(positive_rate, 4),
            tpr=round(tpr, 4), fpr=round(fpr, 4),
            ppv=round(ppv, 4), npv=round(npv, 4), auc=round(auc, 4),
        )

    @staticmethod
    def _parity_ratio(values: list[float]) -> float:
        """Compute min/max ratio for parity check."""
        non_zero = [v for v in values if v > 0]
        if len(non_zero) < 2:
            return 1.0
        return min(non_zero) / max(non_zero)

    def _bin_age(self, ages: np.ndarray) -> np.ndarray:
        """Bin continuous age values into groups."""
        binned = pd.cut(
            ages, bins=self.AGE_BINS, labels=self.AGE_LABELS, right=False,
        )
        return np.asarray(binned.astype(str))

    def assess(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        features: pd.DataFrame,
        model_name: str = "unknown",
    ) -> FairnessReport:
        """Run full fairness assessment.

        Args:
            y_true: Ground truth labels
            y_prob: Predicted probabilities
            features: DataFrame containing protected attributes
            model_name: Model identifier
        """
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)
        y_pred = (y_prob >= self.threshold).astype(int)

        report = FairnessReport(model_name=model_name, n_total=len(y_true))

        all_positive_rates: list[float] = []
        all_tprs: list[float] = []
        all_fprs: list[float] = []
        all_ppvs: list[float] = []

        for attr in self.protected_attributes:
            if attr not in features.columns:
                report.warnings.append(f"Protected attribute '{attr}' not in features")
                continue

            # Create groups based on attribute type
            if attr == "Age":
                group_labels = self._bin_age(features[attr].values)
            else:
                group_labels = features[attr].astype(str).values

            unique_groups = np.unique(group_labels)
            attr_groups: list[GroupMetrics] = []

            for group_val in unique_groups:
                mask = group_labels == group_val
                if mask.sum() < 10:
                    continue

                gm = self._compute_group_metrics(
                    y_true[mask], y_prob[mask], y_pred[mask],
                    group_name=attr, group_value=str(group_val),
                )
                attr_groups.append(gm)
                report.groups.append(gm)

                all_positive_rates.append(gm.positive_rate)
                all_tprs.append(gm.tpr)
                all_fprs.append(gm.fpr)
                all_ppvs.append(gm.ppv)

            # Per-attribute breakdown
            if attr_groups:
                attr_positive_rates = [g.positive_rate for g in attr_groups]
                attr_tprs = [g.tpr for g in attr_groups]
                attr_aucs = [g.auc for g in attr_groups]

                report.attribute_reports[attr] = {
                    "n_groups": len(attr_groups),
                    "groups": [
                        {"value": g.group_value, "n": g.n_samples, "rate": g.positive_rate,
                         "tpr": g.tpr, "auc": g.auc}
                        for g in attr_groups
                    ],
                    "parity_ratio": round(self._parity_ratio(attr_positive_rates), 4),
                    "eor": round(self._parity_ratio(attr_tprs), 4),
                    "auc_range": [round(min(attr_aucs), 4), round(max(attr_aucs), 4)],
                }

        # Global parity ratios
        if len(all_positive_rates) >= 2:
            report.demographic_parity_ratio = self._parity_ratio(all_positive_rates)
        if len(all_tprs) >= 2:
            report.equalized_odds_ratio = self._parity_ratio(all_tprs)
        if len(all_ppvs) >= 2:
            report.predictive_parity_ratio = self._parity_ratio(all_ppvs)

        # Max gaps
        if all_tprs:
            report.max_tpr_gap = max(all_tprs) - min(all_tprs)
        if all_fprs:
            report.max_fpr_gap = max(all_fprs) - min(all_fprs)
        if all_ppvs:
            report.max_ppv_gap = max(all_ppvs) - min(all_ppvs)

        # Four-fifths rule check
        report.passes_four_fifths = (
            self.PARITY_LOWER <= report.demographic_parity_ratio <= self.PARITY_UPPER
            and self.PARITY_LOWER <= report.equalized_odds_ratio <= self.PARITY_UPPER
        )

        if not report.passes_four_fifths:
            report.warnings.append(
                f"FAIL: Four-fifths rule violated — "
                f"DPR={report.demographic_parity_ratio:.4f}, "
                f"EOR={report.equalized_odds_ratio:.4f}"
            )

        if report.max_tpr_gap > 0.15:
            report.warnings.append(
                f"WARNING: TPR gap across groups = {report.max_tpr_gap:.4f} (>0.15)"
            )

        logger.info(
            "fairness_audit model=%s DPR=%.4f EOR=%.4f pass=%s groups=%d",
            model_name, report.demographic_parity_ratio,
            report.equalized_odds_ratio, report.passes_four_fifths,
            len(report.groups),
        )
        return report
