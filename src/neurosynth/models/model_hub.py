# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""NeuroSynth v2 Model Hub — Unified multi-modal prediction interface.

Orchestrates all 5 specialized models and fuses their outputs via
a gradient-boosted meta-learner:

  1. CalibratedEnsemble     — tabular clinical features
  2. BrainConnectomePhase2  — connectome GNN (when imaging data available)
  3. GenomicPhase3           — variant transformer (when genomic data available)
  4. ForecastingPhase4       — TFT temporal forecast (when longitudinal data available)
  5. CausalPhase5            — causal discovery + counterfactual (when DAG available)

The ModelHub provides a single `predict()` entry point that:
  - Dispatches to available models based on input data modalities
  - Fuses outputs via the meta-learner
  - Returns calibrated probabilities with uncertainty bounds
  - Generates per-model explanations
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Modality(StrEnum):
    """Available data modalities for model dispatch."""
    CLINICAL = "clinical"
    CONNECTOME = "connectome"
    GENOMIC = "genomic"
    LONGITUDINAL = "longitudinal"
    CAUSAL = "causal"


@dataclass
class ModelPrediction:
    """Standardized prediction output from any model."""
    model_name: str
    modality: Modality
    probability: float
    uncertainty_lower_80: float = 0.0
    uncertainty_upper_80: float = 1.0
    uncertainty_lower_95: float = 0.0
    uncertainty_upper_95: float = 1.0
    shap_values: list[dict[str, Any]] = field(default_factory=list)
    raw_output: dict[str, Any] = field(default_factory=dict)
    available: bool = True


@dataclass
class FusedPrediction:
    """Final fused prediction from all available models."""
    probability: float
    risk_level: str
    confidence: str
    prediction: int
    model_contributions: dict[str, float] = field(default_factory=dict)
    uncertainty_80: tuple[float, float] = (0.0, 1.0)
    uncertainty_95: tuple[float, float] = (0.0, 1.0)
    per_model: list[ModelPrediction] = field(default_factory=list)
    conformal_set: dict[str, Any] = field(default_factory=dict)
    explanation: dict[str, Any] = field(default_factory=dict)


class ModelHub:
    """Unified multi-modal prediction hub.

    Usage:
        hub = ModelHub(feature_names=["Age", "MMSE", ...])
        hub.register_ensemble(ensemble_model)
        hub.register_connectome(gnn_model)
        hub.register_genomic(genomic_model)

        result = hub.predict(
            clinical_features=np.array([...]),
            connectome_data=None,       # optional
            genomic_data=None,          # optional
            longitudinal_df=None,       # optional
            causal_df=None,             # optional
        )
    """

    # Meta-learner weights (default — can be overridden by training)
    DEFAULT_WEIGHTS = {
        Modality.CLINICAL: 0.40,
        Modality.CONNECTOME: 0.20,
        Modality.GENOMIC: 0.15,
        Modality.LONGITUDINAL: 0.15,
        Modality.CAUSAL: 0.10,
    }

    def __init__(
        self,
        feature_names: list[str],
        decision_threshold: float = 0.5,
    ) -> None:
        self.feature_names = feature_names
        self.decision_threshold = decision_threshold

        # Model registry
        self._ensemble: Any = None
        self._connectome: Any = None
        self._genomic: Any = None
        self._forecasting: Any = None
        self._causal: Any = None

        # Trained meta-learner weights
        self._weights = dict(self.DEFAULT_WEIGHTS)
        self._meta_learner: Any = None

    # ------------------------------------------------------------------
    # Model registration
    # ------------------------------------------------------------------

    def register_ensemble(self, model: Any) -> None:
        """Register the calibrated ensemble model (CalibratedEnsemble or BiomarkerPredictor)."""
        self._ensemble = model
        logger.info("ModelHub: registered ensemble model (%s)", type(model).__name__)

    def register_connectome(self, model: Any) -> None:
        """Register the connectome GNN (BrainConnectomePhase2Model)."""
        self._connectome = model
        logger.info("ModelHub: registered connectome GNN (%s)", type(model).__name__)

    def register_genomic(self, model: Any) -> None:
        """Register the genomic variant transformer (GenomicPhase3Model)."""
        self._genomic = model
        logger.info("ModelHub: registered genomic transformer (%s)", type(model).__name__)

    def register_forecasting(self, model: Any) -> None:
        """Register the TFT forecasting model (ForecastingPhase4Model)."""
        self._forecasting = model
        logger.info("ModelHub: registered TFT forecasting model (%s)", type(model).__name__)

    def register_causal(self, model: Any) -> None:
        """Register the causal engine (CausalPhase5Engine)."""
        self._causal = model
        logger.info("ModelHub: registered causal engine (%s)", type(model).__name__)

    # ------------------------------------------------------------------
    # Model availability check
    # ------------------------------------------------------------------

    @property
    def available_modalities(self) -> list[Modality]:
        """List currently registered and available modalities."""
        available = []
        if self._ensemble is not None:
            available.append(Modality.CLINICAL)
        if self._connectome is not None:
            available.append(Modality.CONNECTOME)
        if self._genomic is not None:
            available.append(Modality.GENOMIC)
        if self._forecasting is not None:
            available.append(Modality.LONGITUDINAL)
        if self._causal is not None:
            available.append(Modality.CAUSAL)
        return available

    # ------------------------------------------------------------------
    # Individual model predictions
    # ------------------------------------------------------------------

    def _predict_clinical(self, X: np.ndarray) -> ModelPrediction:
        """Run the calibrated ensemble on clinical features."""
        if self._ensemble is None:
            return ModelPrediction(
                model_name="ensemble", modality=Modality.CLINICAL,
                probability=0.5, available=False,
            )

        try:
            result = self._ensemble.predict(X)
            prob = float(result.get("probability", 0.5))

            # Extract conformal bounds if available
            conformal = result.get("conformal_prediction", {})

            return ModelPrediction(
                model_name="calibrated_ensemble",
                modality=Modality.CLINICAL,
                probability=prob,
                shap_values=[
                    {"feature": f, "value": 0.0}
                    for f in result.get("top_risk_factors", [])
                ],
                raw_output=result,
                available=True,
            )
        except Exception as e:
            logger.warning("Ensemble prediction failed: %s", e)
            return ModelPrediction(
                model_name="ensemble", modality=Modality.CLINICAL,
                probability=0.5, available=False,
            )

    def _predict_connectome(self, connectome_data: Any) -> ModelPrediction:
        """Run the GNN on connectome data."""
        if self._connectome is None or connectome_data is None:
            return ModelPrediction(
                model_name="connectome_gnn", modality=Modality.CONNECTOME,
                probability=0.5, available=False,
            )

        try:
            result = self._connectome.predict_with_uncertainty(connectome_data)
            return ModelPrediction(
                model_name="connectome_gnn",
                modality=Modality.CONNECTOME,
                probability=float(result.get("mean", 0.5)),
                uncertainty_lower_80=float(result.get("lower_80", 0.0)),
                uncertainty_upper_80=float(result.get("upper_80", 1.0)),
                uncertainty_lower_95=float(result.get("lower_95", 0.0)),
                uncertainty_upper_95=float(result.get("upper_95", 1.0)),
                shap_values=result.get("shap_values", []),
                raw_output=result,
                available=True,
            )
        except Exception as e:
            logger.warning("Connectome prediction failed: %s", e)
            return ModelPrediction(
                model_name="connectome_gnn", modality=Modality.CONNECTOME,
                probability=0.5, available=False,
            )

    def _predict_genomic(self, genomic_data: np.ndarray | None) -> ModelPrediction:
        """Run the genomic transformer on variant data."""
        if self._genomic is None or genomic_data is None:
            return ModelPrediction(
                model_name="genomic_transformer", modality=Modality.GENOMIC,
                probability=0.5, available=False,
            )

        try:
            result = self._genomic.predict_with_uncertainty(genomic_data)
            return ModelPrediction(
                model_name="genomic_transformer",
                modality=Modality.GENOMIC,
                probability=float(result.get("mean", 0.5)),
                uncertainty_lower_80=float(result.get("lower_80", 0.0)),
                uncertainty_upper_80=float(result.get("upper_80", 1.0)),
                uncertainty_lower_95=float(result.get("lower_95", 0.0)),
                uncertainty_upper_95=float(result.get("upper_95", 1.0)),
                shap_values=result.get("shap_values", []),
                raw_output=result,
                available=True,
            )
        except Exception as e:
            logger.warning("Genomic prediction failed: %s", e)
            return ModelPrediction(
                model_name="genomic_transformer", modality=Modality.GENOMIC,
                probability=0.5, available=False,
            )

    def _predict_longitudinal(self, longitudinal_df: pd.DataFrame | None) -> ModelPrediction:
        """Run the TFT on longitudinal data."""
        if self._forecasting is None or longitudinal_df is None:
            return ModelPrediction(
                model_name="tft_forecasting", modality=Modality.LONGITUDINAL,
                probability=0.5, available=False,
            )

        try:
            result = self._forecasting.predict_with_uncertainty(longitudinal_df)
            # TFT returns a trajectory; take the endpoint as summary probability
            mean_traj = result.get("mean", [0.5])
            endpoint_prob = float(mean_traj[-1]) if isinstance(mean_traj, list) else float(mean_traj)

            return ModelPrediction(
                model_name="tft_forecasting",
                modality=Modality.LONGITUDINAL,
                probability=endpoint_prob,
                shap_values=result.get("shap_values", []),
                raw_output=result,
                available=True,
            )
        except Exception as e:
            logger.warning("TFT prediction failed: %s", e)
            return ModelPrediction(
                model_name="tft_forecasting", modality=Modality.LONGITUDINAL,
                probability=0.5, available=False,
            )

    def _predict_causal(
        self, causal_df: pd.DataFrame | None, target: str = "Diagnosis"
    ) -> ModelPrediction:
        """Run the causal engine for intervention effect estimation."""
        if self._causal is None or causal_df is None:
            return ModelPrediction(
                model_name="causal_engine", modality=Modality.CAUSAL,
                probability=0.5, available=False,
            )

        try:
            # Use bootstrap prediction with uncertainty
            y = causal_df[target] if target in causal_df.columns else pd.Series(np.zeros(len(causal_df)))
            X = causal_df.drop(columns=[target], errors="ignore")
            result = self._causal.predict_with_uncertainty(X, y)

            mean_pred = result.get("mean", [0.5])
            avg_prob = float(np.mean(mean_pred)) if isinstance(mean_pred, list) else float(mean_pred)

            return ModelPrediction(
                model_name="causal_engine",
                modality=Modality.CAUSAL,
                probability=np.clip(avg_prob, 0.0, 1.0),
                shap_values=result.get("shap_values", []),
                raw_output=result,
                available=True,
            )
        except Exception as e:
            logger.warning("Causal prediction failed: %s", e)
            return ModelPrediction(
                model_name="causal_engine", modality=Modality.CAUSAL,
                probability=0.5, available=False,
            )

    # ------------------------------------------------------------------
    # Meta-learner fusion
    # ------------------------------------------------------------------

    def train_meta_learner(
        self,
        model_predictions: list[list[ModelPrediction]],
        targets: np.ndarray,
    ) -> dict[str, float]:
        """Train a gradient-boosted meta-learner on model outputs.

        Args:
            model_predictions: List of per-sample prediction lists
            targets: Ground truth labels
        """
        from sklearn.ensemble import GradientBoostingClassifier

        # Build feature matrix from model predictions
        n_samples = len(model_predictions)
        n_models = len(Modality)
        X_meta = np.full((n_samples, n_models), 0.5)

        for i, preds in enumerate(model_predictions):
            for pred in preds:
                modality_idx = list(Modality).index(pred.modality)
                if pred.available:
                    X_meta[i, modality_idx] = pred.probability

        self._meta_learner = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42,
        )
        self._meta_learner.fit(X_meta, targets)

        # Extract learned weights
        importances = self._meta_learner.feature_importances_
        for modality, importance in zip(Modality, importances):
            self._weights[modality] = float(importance)

        probs = self._meta_learner.predict_proba(X_meta)[:, 1]
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(targets, probs)) if len(np.unique(targets)) > 1 else 0.5

        metrics = {
            "meta_auc": round(auc, 4),
            **{f"weight_{m.value}": round(w, 4) for m, w in self._weights.items()},
        }
        logger.info("Meta-learner trained: %s", metrics)
        return metrics

    def _fuse_predictions(self, predictions: list[ModelPrediction]) -> float:
        """Fuse individual model predictions into a single probability."""
        if self._meta_learner is not None:
            # Use trained meta-learner
            X = np.full((1, len(Modality)), 0.5)
            for pred in predictions:
                idx = list(Modality).index(pred.modality)
                if pred.available:
                    X[0, idx] = pred.probability
            return float(self._meta_learner.predict_proba(X)[0, 1])

        # Fallback: weighted average of available models
        total_weight = 0.0
        weighted_sum = 0.0

        for pred in predictions:
            if pred.available:
                w = self._weights.get(pred.modality, 0.1)
                weighted_sum += w * pred.probability
                total_weight += w

        if total_weight == 0:
            return 0.5

        return weighted_sum / total_weight

    # ------------------------------------------------------------------
    # Unified prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        clinical_features: np.ndarray,
        connectome_data: Any = None,
        genomic_data: np.ndarray | None = None,
        longitudinal_df: pd.DataFrame | None = None,
        causal_df: pd.DataFrame | None = None,
    ) -> FusedPrediction:
        """Run all available models and return a fused prediction.

        Only models with available data are executed. Missing modalities
        are excluded from the fusion.
        """
        # Run all models
        predictions = [
            self._predict_clinical(clinical_features),
            self._predict_connectome(connectome_data),
            self._predict_genomic(genomic_data),
            self._predict_longitudinal(longitudinal_df),
            self._predict_causal(causal_df),
        ]

        # Filter to available predictions for fusion
        available = [p for p in predictions if p.available]
        logger.info(
            "ModelHub: %d/%d models available for fusion: %s",
            len(available), len(predictions),
            [p.model_name for p in available],
        )

        # Fuse predictions
        fused_prob = self._fuse_predictions(predictions)
        fused_prob = float(np.clip(fused_prob, 0.0, 1.0))

        # Risk level
        if fused_prob >= 0.8:
            risk_level = "Critical"
        elif fused_prob >= 0.65:
            risk_level = "High"
        elif fused_prob >= 0.4:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        confidence = "High" if abs(fused_prob - 0.5) >= 0.3 else (
            "Medium" if abs(fused_prob - 0.5) >= 0.15 else "Low"
        )

        # Aggregate uncertainty bounds
        if available:
            lower_80 = float(np.mean([p.uncertainty_lower_80 for p in available]))
            upper_80 = float(np.mean([p.uncertainty_upper_80 for p in available]))
            lower_95 = float(np.mean([p.uncertainty_lower_95 for p in available]))
            upper_95 = float(np.mean([p.uncertainty_upper_95 for p in available]))
        else:
            lower_80, upper_80 = 0.0, 1.0
            lower_95, upper_95 = 0.0, 1.0

        # Model contributions
        contributions = {}
        for pred in available:
            contributions[pred.model_name] = round(
                self._weights.get(pred.modality, 0.1), 4
            )

        # Aggregate explanations
        all_shap: dict[str, float] = {}
        for pred in available:
            for sv in pred.shap_values:
                feat = sv.get("feature") or sv.get("region_index", "unknown")
                key = f"{pred.model_name}:{feat}"
                all_shap[key] = float(sv.get("value", 0.0))

        return FusedPrediction(
            probability=round(fused_prob, 4),
            risk_level=risk_level,
            confidence=confidence,
            prediction=int(fused_prob >= self.decision_threshold),
            model_contributions=contributions,
            uncertainty_80=(round(lower_80, 4), round(upper_80, 4)),
            uncertainty_95=(round(lower_95, 4), round(upper_95, 4)),
            per_model=predictions,
            explanation={"top_features": dict(sorted(all_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:10])},
        )
