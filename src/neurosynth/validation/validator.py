"""Core model validation suite for NeuroSynth v2.

Computes comprehensive model evaluation metrics:
  - Classification: AUC, F1, precision, recall, balanced accuracy
  - Calibration: Expected Calibration Error (ECE), Brier score, reliability diagram data
  - Discrimination: sensitivity/specificity at optimal threshold
  - Stability: SHAP top-k consistency across bootstrap seeds
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Expected Calibration Error and reliability diagram data."""
    ece: float = 0.0
    mce: float = 0.0  # Maximum calibration error
    brier: float = 0.0
    bin_confidences: list[float] = field(default_factory=list)
    bin_accuracies: list[float] = field(default_factory=list)
    bin_counts: list[int] = field(default_factory=list)
    n_bins: int = 15


@dataclass
class ValidationReport:
    """Full validation report for a model on a single disease/task."""
    model_name: str
    disease: str
    n_samples: int

    # Classification metrics
    auc: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    specificity: float = 0.0
    logloss: float = 0.0

    # Calibration
    calibration: CalibrationMetrics = field(default_factory=CalibrationMetrics)

    # Threshold
    optimal_threshold: float = 0.5
    sensitivity_at_threshold: float = 0.0
    specificity_at_threshold: float = 0.0

    # Stability
    shap_top5_jaccard: float = 0.0
    shap_stability_seeds: int = 0

    # ROC curve data (for plotting)
    fpr: list[float] = field(default_factory=list)
    tpr: list[float] = field(default_factory=list)

    # PR curve data
    pr_precisions: list[float] = field(default_factory=list)
    pr_recalls: list[float] = field(default_factory=list)

    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "disease": self.disease,
            "n_samples": self.n_samples,
            "auc": round(self.auc, 4),
            "f1": round(self.f1, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "accuracy": round(self.accuracy, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "specificity": round(self.specificity, 4),
            "logloss": round(self.logloss, 4),
            "ece": round(self.calibration.ece, 4),
            "mce": round(self.calibration.mce, 4),
            "brier": round(self.calibration.brier, 4),
            "optimal_threshold": round(self.optimal_threshold, 4),
            "shap_top5_jaccard": round(self.shap_top5_jaccard, 4),
            "warnings": self.warnings,
        }


class ModelValidator:
    """Comprehensive model validation engine.

    Usage:
        validator = ModelValidator()
        report = validator.validate(
            y_true=test_labels,
            y_prob=predicted_probabilities,
            model_name="calibrated_ensemble",
            disease="Alzheimer's Disease",
        )
    """

    def __init__(self, n_calibration_bins: int = 15) -> None:
        self.n_calibration_bins = n_calibration_bins

    # ------------------------------------------------------------------
    # Expected Calibration Error (ECE)
    # ------------------------------------------------------------------

    def compute_ece(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int | None = None,
    ) -> CalibrationMetrics:
        """Compute ECE (Expected Calibration Error).

        ECE = Σ (|B_m| / n) × |acc(B_m) - conf(B_m)|

        where B_m is the set of samples in bin m, acc is accuracy,
        and conf is mean predicted probability.
        """
        n_bins = n_bins or self.n_calibration_bins
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []

        ece = 0.0
        mce = 0.0
        n = len(y_true)

        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            if i == n_bins - 1:
                mask = (y_prob >= lo) & (y_prob <= hi)
            else:
                mask = (y_prob >= lo) & (y_prob < hi)

            count = int(mask.sum())
            if count == 0:
                bin_confidences.append(0.0)
                bin_accuracies.append(0.0)
                bin_counts.append(0)
                continue

            acc = float(y_true[mask].mean())
            conf = float(y_prob[mask].mean())
            gap = abs(acc - conf)

            ece += (count / n) * gap
            mce = max(mce, gap)

            bin_confidences.append(round(conf, 4))
            bin_accuracies.append(round(acc, 4))
            bin_counts.append(count)

        return CalibrationMetrics(
            ece=round(ece, 6),
            mce=round(mce, 6),
            brier=round(float(brier_score_loss(y_true, y_prob)), 6),
            bin_confidences=bin_confidences,
            bin_accuracies=bin_accuracies,
            bin_counts=bin_counts,
            n_bins=n_bins,
        )

    # ------------------------------------------------------------------
    # Optimal threshold (Youden's J)
    # ------------------------------------------------------------------

    @staticmethod
    def find_optimal_threshold(
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> tuple[float, float, float]:
        """Find optimal threshold via Youden's J statistic.

        Returns (threshold, sensitivity, specificity).
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))

        threshold = float(thresholds[best_idx])
        sensitivity = float(tpr[best_idx])
        specificity = float(1.0 - fpr[best_idx])

        return threshold, sensitivity, specificity

    # ------------------------------------------------------------------
    # SHAP stability
    # ------------------------------------------------------------------

    @staticmethod
    def shap_stability(
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
        n_seeds: int = 5,
        top_k: int = 5,
    ) -> float:
        """Measure stability of SHAP top-k features across bootstrap seeds.

        Returns mean pairwise Jaccard similarity (1.0 = perfectly stable).
        """
        try:
            import shap
# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
        except ImportError:
            return 0.0

        top_k_sets: list[set[str]] = []
        rng = np.random.RandomState(42)

        for seed in range(n_seeds):
            idx = rng.choice(len(X), size=min(200, len(X)), replace=True)
            X_sample = X[idx]

            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)

                if isinstance(shap_values, list):
                    vals = np.asarray(shap_values[1] if len(shap_values) == 2 else shap_values[0])
                else:
                    vals = np.asarray(shap_values)
                    if vals.ndim == 3:
                        vals = vals[:, :, 1] if vals.shape[-1] == 2 else vals.mean(axis=-1)

                mean_abs = np.mean(np.abs(vals), axis=0)
                top_indices = np.argsort(mean_abs)[::-1][:top_k]
                top_features = {feature_names[i] for i in top_indices}
                top_k_sets.append(top_features)
            except Exception:
                continue

        if len(top_k_sets) < 2:
            return 0.0

        # Pairwise Jaccard
        jaccards = []
        for i in range(len(top_k_sets)):
            for j in range(i + 1, len(top_k_sets)):
                inter = len(top_k_sets[i] & top_k_sets[j])
                union = len(top_k_sets[i] | top_k_sets[j])
                jaccards.append(inter / union if union > 0 else 0.0)

        return float(np.mean(jaccards))

    # ------------------------------------------------------------------
    # Full validation
    # ------------------------------------------------------------------

    def validate(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "unknown",
        disease: str = "all",
        model: Any = None,
        X: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> ValidationReport:
        """Run full validation suite on predictions.

        Args:
            y_true: Ground truth binary labels
            y_prob: Predicted probabilities
            model_name: Name identifier for the model
            disease: Disease context
            model: (Optional) trained model for SHAP stability analysis
            X: (Optional) feature matrix for SHAP stability
            feature_names: (Optional) feature names for SHAP stability
        """
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)

        # Optimal threshold
        threshold, sensitivity, specificity = self.find_optimal_threshold(y_true, y_prob)
        y_pred = (y_prob >= threshold).astype(int)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)

        # PR curve
        pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_prob)

        # Calibration
        calibration = self.compute_ece(y_true, y_prob)

        # SHAP stability
        shap_jaccard = 0.0
        shap_seeds = 0
        if model is not None and X is not None and feature_names is not None:
            shap_jaccard = self.shap_stability(model, X, feature_names, n_seeds=5, top_k=5)
            shap_seeds = 5

        report = ValidationReport(
            model_name=model_name,
            disease=disease,
            n_samples=len(y_true),
            auc=round(float(roc_auc_score(y_true, y_prob)), 4),
            f1=round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            precision=round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            recall=round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            accuracy=round(float(accuracy_score(y_true, y_pred)), 4),
            balanced_accuracy=round(float(balanced_accuracy_score(y_true, y_pred)), 4),
            specificity=round(specificity, 4),
            logloss=round(float(log_loss(y_true, y_prob)), 4),
            calibration=calibration,
            optimal_threshold=round(threshold, 4),
            sensitivity_at_threshold=round(sensitivity, 4),
            specificity_at_threshold=round(specificity, 4),
            shap_top5_jaccard=round(shap_jaccard, 4),
            shap_stability_seeds=shap_seeds,
            fpr=[round(float(x), 4) for x in fpr.tolist()],
            tpr=[round(float(x), 4) for x in tpr.tolist()],
            pr_precisions=[round(float(x), 4) for x in pr_prec.tolist()],
            pr_recalls=[round(float(x), 4) for x in pr_rec.tolist()],
        )

        logger.info(
            "validation_complete model=%s disease=%s auc=%.4f ece=%.4f brier=%.4f",
            model_name, disease, report.auc, calibration.ece, calibration.brier,
        )
        return report
