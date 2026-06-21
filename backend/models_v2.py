# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""v2 Enhanced Pydantic models for Priority 5 — Inference API Refactor.

Adds structured response models for:
  - LIME explanations
  - Counterfactual recommendations
  - Conformal confidence intervals
  - Causal interventions
  - Extended 48-month trajectory
  - RFC 7807 error detail
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SHAPValue(BaseModel):
    """Single SHAP feature attribution."""
    feature: str
    value: float


class LIMEExplanation(BaseModel):
    """LIME-style local explanation."""
    feature: str
    weight: float
    direction: str = ""  # "increases_risk" or "decreases_risk"


class Counterfactual(BaseModel):
    """A counterfactual recommendation (what-if)."""
    feature: str
    current_value: float
    target_value: float
    risk_delta: float
    interpretation: str = ""


class CausalIntervention(BaseModel):
    """Causal intervention effect estimate."""
    factor: str
    effect_size: float
    direction: str = ""  # "protective" or "amplifying"
    confidence: float = 0.0


class ConfidenceInterval(BaseModel):
    """Calibrated confidence interval for a probability."""
    method: str = "conformal"
    coverage: float = 0.95
    lower: float = 0.0
    upper: float = 1.0


class DiseaseProb(BaseModel):
    """Probability with confidence interval for a single disease."""
    probability: float
    ci_lower: float = 0.0
    ci_upper: float = 1.0


class TrajectoryForecast(BaseModel):
    """Extended trajectory forecast (up to 48 months)."""
    months: list[int] = Field(default_factory=lambda: [6, 12, 18, 24, 30, 36, 42, 48])
    values: list[float] = Field(default_factory=list)
    bands_lower: list[float] = Field(default_factory=list)
    bands_upper: list[float] = Field(default_factory=list)


class ModelContribution(BaseModel):
    """Individual model contribution to the fused prediction."""
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    probability: float
    weight: float = 0.0


class AnalyzeResponseV2(BaseModel):
    """Enhanced v2 analysis response."""
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "patient_id": "P-001",
                "request_id": "abc-123",
                "prediction": 0,
                "probability": 0.32,
                "risk_level": "Low",
                "confidence": "High",
            }
        }
    )

    # Core prediction
    patient_id: str
    request_id: str
    prediction: int
    probability: float
    risk_level: str
    confidence: str

    # Per-disease probabilities with CIs
    disease_probabilities: dict[str, DiseaseProb] = Field(default_factory=dict)

    # Per-model breakdown
    model_contributions: list[ModelContribution] = Field(default_factory=list)

    # Explainability
    shap_values: list[SHAPValue] = Field(default_factory=list)
    lime_explanation: list[LIMEExplanation] = Field(default_factory=list)
    counterfactuals: list[Counterfactual] = Field(default_factory=list)
    top_risk_factors: list[str] = Field(default_factory=list)

    # Trajectory (extended to 48 months)
    trajectory_48mo: TrajectoryForecast = Field(default_factory=TrajectoryForecast)

    # Causal
    causal_interventions: list[CausalIntervention] = Field(default_factory=list)
    causal_graph: dict[str, Any] = Field(default_factory=dict)

    # Confidence intervals
    confidence_intervals: ConfidenceInterval = Field(
        default_factory=lambda: ConfidenceInterval(method="conformal", coverage=0.95)
    )

    # Clinical report
    report_text: str = ""
    report: dict[str, Any] = Field(default_factory=dict)

    # Disease classification
    disease_classification: dict[str, Any] = Field(default_factory=dict)

    # Metadata
    individual_model_probs: dict[str, float] = Field(default_factory=dict)
    timestamp: str = ""
    api_version: str = "v2"


class RFC7807Error(BaseModel):
    """RFC 7807 Problem Details for HTTP APIs."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "https://neurosynth.dev/errors/validation",
                "title": "Validation Error",
                "status": 422,
                "detail": "MMSE must be between 0 and 30",
                "instance": "/predictions/analyze",
            }
        }
    )
    type: str = "about:blank"
    title: str
    status: int
    detail: str
    instance: str = ""
    errors: list[dict[str, Any]] = Field(default_factory=list)
    trace_id: str = ""
