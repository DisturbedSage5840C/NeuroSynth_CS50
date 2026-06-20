"""v3 Pydantic response models for Part 4 — Backend Changes.

NOTE: Do NOT add ``from __future__ import annotations``.
FastAPI/Pydantic requires runtime type resolution for route parameters.
"""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from backend.models_v2 import (
# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
    AnalyzeResponseV2,
    CausalIntervention,
    ConfidenceInterval,
    Counterfactual,
    DiseaseProb,
    LIMEExplanation,
    ModelContribution,
    SHAPValue,
    TrajectoryForecast,
)


# ── Data source models ───────────────────────────────────────────────────────

class DataSourceStatus(BaseModel):
    """Status of a single real-data source."""
    name: str
    display_name: str
    tier: str                  # "1", "2", "3", "—" (synthetic)
    url: str | None = None
    row_count: int | None = None
    feature_count: int | None = None
    last_updated: datetime | None = None
    status: str = "active"     # "active", "pending", "error"
    features: str = ""         # human-readable feature summary
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataSourcesResponse(BaseModel):
    """Response for GET /v3/data/sources."""
    sources: list[DataSourceStatus]
    total_rows: int = 0
    total_sources: int = 0
    active_sources: int = 0
    last_refreshed: datetime | None = None


class RefreshResponse(BaseModel):
    """Response for POST /v3/data/refresh/{source}."""
    source: str
    status: str
    message: str
    triggered_at: datetime = Field(default_factory=datetime.utcnow)


# ── Cohort statistics ────────────────────────────────────────────────────────

class DiseasePrevalence(BaseModel):
    name: str
    value: float       # percentage
    count: int = 0
    color: str = ""


class AgeGroup(BaseModel):
    range: str         # "20–30", "30–40", …
    ad: int = 0
    pd: int = 0
    ms: int = 0
    ep: int = 0
    als: int = 0
    hd: int = 0


class CohortStatsResponse(BaseModel):
    """Response for GET /v3/data/cohort/stats."""
    total_patients: int
    data_sources: int
    prevalence: list[DiseasePrevalence]
    age_distribution: list[AgeGroup]
    feature_count: int = 56
    schema_version: str = "v5"
    computed_at: datetime | None = None


# ── Data provenance ──────────────────────────────────────────────────────────

class ProvenanceRow(BaseModel):
    source: str
    tier: str
    rows_raw: int
    rows_after_qc: int
    features_mapped: int
    synthetic: bool = False


class ProvenanceResponse(BaseModel):
    """Response for GET /v3/data/provenance."""
    total_rows: int
    provenance: list[ProvenanceRow]
    merge_file: str = "data/real_v5.parquet"
    schema_version: str = "v5"
    merged_at: datetime | None = None


# ── Fusion weights ───────────────────────────────────────────────────────────

class FusionWeightsResponse(BaseModel):
    """Response for GET /v3/fusion/weights."""
    weights: dict[str, float]
    method: str = "default"    # "optuna" | "default" | "cross_attention"
    val_auc: float | None = None
    trial: int | None = None
    updated_at: datetime | None = None


# ── v3 Analyze response ──────────────────────────────────────────────────────

class ModalityContribution(BaseModel):
    """Per-modality probability and attention weight."""
    modality: str
    probability: float
    weight: float = 0.0
    attention_weight: float = 0.0


class AnalyzeResponseV3(BaseModel):
    """v3 analysis: all v2 fields + cross-attention fusion + RAG metadata."""
    model_config = ConfigDict(protected_namespaces=())

    # ── Core (mirrors AnalyzeResponseV2) ──
    patient_id: str
    request_id: str
    prediction: int
    probability: float
    risk_level: str
    confidence: str

    disease_probabilities: dict[str, DiseaseProb] = Field(default_factory=dict)
    model_contributions: list[ModelContribution] = Field(default_factory=list)

    shap_values: list[SHAPValue] = Field(default_factory=list)
    lime_explanations: list[LIMEExplanation] = Field(default_factory=list)
    counterfactuals: list[Counterfactual] = Field(default_factory=list)
    causal_interventions: list[CausalIntervention] = Field(default_factory=list)

    conformal_interval: ConfidenceInterval = Field(
        default_factory=lambda: ConfidenceInterval()
    )
    trajectory: TrajectoryForecast = Field(default_factory=TrajectoryForecast)

    causal_analysis: dict[str, Any] = Field(default_factory=dict)
    disease_classification: dict[str, Any] = Field(default_factory=dict)

    clinical_report: dict[str, Any] | None = None
    generated_by: str = "v3"

    latency_ms: float = 0.0

    # ── v3-specific ──
    fusion_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "tabular": 0.40, "genomic": 0.15, "tft": 0.15, "causal": 0.10, "gnn": 0.20,
        }
    )
    fusion_attention_map: list[float] = Field(default_factory=list)
    modality_contributions: list[ModalityContribution] = Field(default_factory=list)

    rag_citations: list[str] = Field(default_factory=list)
    rag_docs_retrieved: int = 0
    schema_version: str = "v3"
