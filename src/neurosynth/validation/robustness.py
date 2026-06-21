# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""Robustness testing for NeuroSynth v2 models.

Adversarial perturbation tests to evaluate model stability:
  - Gaussian noise injection (feature-level)
  - Feature dropout (random masking)
  - Covariate shift simulation
  - Boundary sample analysis (near-threshold patients)
  - Label noise robustness

Follows FDA SaMD Pre-Submissions guidance for worst-case testing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


@dataclass
class PerturbationResult:
    """Result of a single perturbation test."""
    test_name: str
    original_auc: float
    perturbed_auc: float
    auc_delta: float
    original_predictions: list[float] = field(default_factory=list)
    perturbed_predictions: list[float] = field(default_factory=list)
    max_prediction_shift: float = 0.0
    mean_prediction_shift: float = 0.0
    passed: bool = True
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RobustnessReport:
    """Full robustness assessment."""
    model_name: str
    n_samples: int
    tests: list[PerturbationResult] = field(default_factory=list)
    overall_pass: bool = True
    worst_auc_drop: float = 0.0
    worst_test: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "n_samples": self.n_samples,
            "n_tests": len(self.tests),
            "overall_pass": self.overall_pass,
            "worst_auc_drop": round(self.worst_auc_drop, 4),
            "worst_test": self.worst_test,
            "tests": [
                {
                    "name": t.test_name,
                    "original_auc": round(t.original_auc, 4),
                    "perturbed_auc": round(t.perturbed_auc, 4),
                    "auc_delta": round(t.auc_delta, 4),
                    "max_pred_shift": round(t.max_prediction_shift, 4),
                    "mean_pred_shift": round(t.mean_prediction_shift, 4),
                    "passed": t.passed,
                }
                for t in self.tests
            ],
            "warnings": self.warnings,
        }


class RobustnessTester:
    """Adversarial robustness evaluation for clinical models.

    Tests model stability under realistic perturbations that
    could occur in clinical deployment (measurement noise,
    missing data, population shift).
    """

    # Maximum acceptable AUC drop for each perturbation level
    AUC_DROP_THRESHOLD = 0.05          # 5% AUC drop → FAIL
    PREDICTION_SHIFT_THRESHOLD = 0.15  # 15% max prediction shift → WARNING

    def __init__(self, random_state: int = 42) -> None:
        self.rng = np.random.RandomState(random_state)

    def _evaluate_perturbation(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X_original: np.ndarray,
        X_perturbed: np.ndarray,
        y_true: np.ndarray,
        test_name: str,
    ) -> PerturbationResult:
        """Compare model predictions between original and perturbed inputs."""
        orig_probs = predict_fn(X_original)
        pert_probs = predict_fn(X_perturbed)

        orig_auc = float(roc_auc_score(y_true, orig_probs)) if len(np.unique(y_true)) > 1 else 0.5
        pert_auc = float(roc_auc_score(y_true, pert_probs)) if len(np.unique(y_true)) > 1 else 0.5
        auc_delta = pert_auc - orig_auc

        shifts = np.abs(pert_probs - orig_probs)
        max_shift = float(shifts.max())
        mean_shift = float(shifts.mean())

        passed = abs(auc_delta) < self.AUC_DROP_THRESHOLD

        return PerturbationResult(
            test_name=test_name,
            original_auc=orig_auc,
            perturbed_auc=pert_auc,
            auc_delta=round(auc_delta, 6),
            original_predictions=orig_probs[:20].tolist(),
            perturbed_predictions=pert_probs[:20].tolist(),
            max_prediction_shift=max_shift,
            mean_prediction_shift=mean_shift,
            passed=passed,
        )

    # ------------------------------------------------------------------
    # Perturbation tests
    # ------------------------------------------------------------------

    def gaussian_noise(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X: np.ndarray,
        y: np.ndarray,
        noise_scale: float = 0.05,
    ) -> PerturbationResult:
        """Inject Gaussian noise at a fraction of feature standard deviation."""
        stds = np.std(X, axis=0, keepdims=True)
        stds = np.where(stds == 0, 1.0, stds)
        noise = self.rng.normal(0, noise_scale, X.shape) * stds
        X_noisy = X + noise

        result = self._evaluate_perturbation(
            predict_fn, X, X_noisy, y,
            test_name=f"gaussian_noise_{noise_scale}",
        )
        result.details = {"noise_scale": noise_scale}
        return result

    def feature_dropout(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X: np.ndarray,
        y: np.ndarray,
        dropout_rate: float = 0.10,
    ) -> PerturbationResult:
        """Randomly mask features with column medians (simulating missing data)."""
        medians = np.median(X, axis=0)
        mask = self.rng.random(X.shape) < dropout_rate
        X_dropped = X.copy()
        for j in range(X.shape[1]):
            X_dropped[mask[:, j], j] = medians[j]

        result = self._evaluate_perturbation(
            predict_fn, X, X_dropped, y,
            test_name=f"feature_dropout_{dropout_rate}",
        )
        result.details = {"dropout_rate": dropout_rate, "features_dropped": int(mask.sum())}
        return result

    def covariate_shift(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X: np.ndarray,
        y: np.ndarray,
        shift_features: list[int] | None = None,
        shift_magnitude: float = 1.0,
    ) -> PerturbationResult:
        """Simulate population-level covariate shift (e.g., older cohort)."""
        X_shifted = X.copy()
        features = shift_features or list(range(min(5, X.shape[1])))

        for feat_idx in features:
            feat_std = np.std(X[:, feat_idx])
            if feat_std > 0:
                X_shifted[:, feat_idx] += shift_magnitude * feat_std

        result = self._evaluate_perturbation(
            predict_fn, X, X_shifted, y,
            test_name=f"covariate_shift_{shift_magnitude}",
        )
        result.details = {"shift_magnitude": shift_magnitude, "shifted_features": features}
        return result

    def boundary_analysis(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.5,
        margin: float = 0.1,
    ) -> PerturbationResult:
        """Analyze model stability for samples near the decision boundary."""
        probs = predict_fn(X)
        boundary_mask = np.abs(probs - threshold) < margin
        n_boundary = int(boundary_mask.sum())

        if n_boundary < 10:
            return PerturbationResult(
                test_name="boundary_analysis",
                original_auc=0.0, perturbed_auc=0.0, auc_delta=0.0,
                passed=True,
                details={"n_boundary": n_boundary, "margin": margin, "skip": True},
            )

        # Small perturbation to boundary samples
        X_boundary = X[boundary_mask]
        y_boundary = y[boundary_mask]
        noise = self.rng.normal(0, 0.02, X_boundary.shape) * np.std(X, axis=0, keepdims=True)
        X_perturbed = X_boundary + noise

        orig_preds = predict_fn(X_boundary)
        pert_preds = predict_fn(X_perturbed)

        # How many flipped?
        orig_labels = (orig_preds >= threshold).astype(int)
        pert_labels = (pert_preds >= threshold).astype(int)
        n_flipped = int((orig_labels != pert_labels).sum())
        flip_rate = n_flipped / max(n_boundary, 1)

        return PerturbationResult(
            test_name="boundary_analysis",
            original_auc=0.0,
            perturbed_auc=0.0,
            auc_delta=0.0,
            max_prediction_shift=float(np.max(np.abs(pert_preds - orig_preds))),
            mean_prediction_shift=float(np.mean(np.abs(pert_preds - orig_preds))),
            passed=flip_rate < 0.3,  # <30% of boundary samples should flip
            details={
                "n_boundary": n_boundary,
                "n_flipped": n_flipped,
                "flip_rate": round(flip_rate, 4),
                "margin": margin,
            },
        )

    def label_noise(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X: np.ndarray,
        y: np.ndarray,
        noise_rate: float = 0.05,
    ) -> PerturbationResult:
        """Test model performance against noisy labels (simulating annotation errors)."""
        y_noisy = y.copy()
        flip_mask = self.rng.random(len(y)) < noise_rate
        y_noisy[flip_mask] = 1 - y_noisy[flip_mask]

        probs = predict_fn(X)
        orig_auc = float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else 0.5
        noisy_auc = float(roc_auc_score(y_noisy, probs)) if len(np.unique(y_noisy)) > 1 else 0.5

        return PerturbationResult(
            test_name=f"label_noise_{noise_rate}",
            original_auc=orig_auc,
            perturbed_auc=noisy_auc,
            auc_delta=round(noisy_auc - orig_auc, 6),
            passed=abs(noisy_auc - orig_auc) < 0.10,
            details={"noise_rate": noise_rate, "n_flipped": int(flip_mask.sum())},
        )

    # ------------------------------------------------------------------
    # Full robustness suite
    # ------------------------------------------------------------------

    def run_all(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = "unknown",
    ) -> RobustnessReport:
        """Run the complete robustness test suite."""
        report = RobustnessReport(model_name=model_name, n_samples=len(X))

        tests = [
            self.gaussian_noise(predict_fn, X, y, noise_scale=0.03),
            self.gaussian_noise(predict_fn, X, y, noise_scale=0.05),
            self.gaussian_noise(predict_fn, X, y, noise_scale=0.10),
            self.feature_dropout(predict_fn, X, y, dropout_rate=0.05),
            self.feature_dropout(predict_fn, X, y, dropout_rate=0.10),
            self.feature_dropout(predict_fn, X, y, dropout_rate=0.20),
            self.covariate_shift(predict_fn, X, y, shift_magnitude=0.5),
            self.covariate_shift(predict_fn, X, y, shift_magnitude=1.0),
            self.boundary_analysis(predict_fn, X, y),
            self.label_noise(predict_fn, X, y, noise_rate=0.05),
        ]

        report.tests = tests

        # Determine worst case and overall pass
        for test in tests:
            if not test.passed:
                report.overall_pass = False
                report.warnings.append(f"FAIL: {test.test_name} (AUC Δ={test.auc_delta:.4f})")

            drop = abs(test.auc_delta)
            if drop > report.worst_auc_drop:
                report.worst_auc_drop = drop
                report.worst_test = test.test_name

            if test.max_prediction_shift > self.PREDICTION_SHIFT_THRESHOLD:
                report.warnings.append(
                    f"WARNING: {test.test_name} max prediction shift "
                    f"= {test.max_prediction_shift:.4f}"
                )

        logger.info(
            "robustness_complete model=%s tests=%d pass=%s worst_drop=%.4f",
            model_name, len(tests), report.overall_pass, report.worst_auc_drop,
        )
        return report
