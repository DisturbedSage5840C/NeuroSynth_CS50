from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

class PatientSummary(BaseModel):
    disease_stage: str
    progression_category: Literal["slow", "moderate", "rapid"]
    primary_biomarker_pattern: str


class DeteriorationForecast(BaseModel):
    horizon_months: list[int]
    dci_median: list[float]
    dci_ci_80_lower: list[float]
    dci_ci_80_upper: list[float]
    months_to_clinical_threshold: dict[str, float | list[float]]
    forecast_confidence: Literal["low", "moderate", "high"]
    confidence_rationale: str


class PrimaryDriver(BaseModel):
    variable: str
    causal_effect_on_dci: float
    mechanistic_explanation: str


class SecondaryDriver(BaseModel):
    variable: str
    causal_effect: float
    explanation: str


class CausalAnalysis(BaseModel):
    primary_driver: PrimaryDriver
    secondary_drivers: list[SecondaryDriver]
    causal_pathway_narrative: str


class InterventionRecommendation(BaseModel):
    rank: int
    target_variable: str
    intervention_description: str
    estimated_dci_reduction_24mo: float
    estimated_reduction_ci_80: list[float]
    mechanism: str
    evidence_strength: Literal["observational", "RCT_phase2", "RCT_phase3", "meta_analysis"]
    supporting_pmids: list[str]
    contraindications: list[str]
    monitoring_parameters: list[str]


class MonitoringProtocol(BaseModel):
    recommended_biomarkers: list[dict]
    red_flag_thresholds: list[dict]
    next_review_months: int


class ReportSchema(BaseModel):
    report_id: str
    generated_at: datetime
    patient_summary: PatientSummary
    deterioration_forecast: DeteriorationForecast
    causal_analysis: CausalAnalysis
    intervention_recommendations: list[InterventionRecommendation]
    monitoring_protocol: MonitoringProtocol
    uncertainty_flags: list[str] = Field(min_length=1)
    disclaimer: str
