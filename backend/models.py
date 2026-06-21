# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from backend.core.security import Role


class ApiMessage(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"message": "ok"}})
    message: str = Field(description="Human-readable API message")


class UserContext(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "doctor.jones",
                "role": "CLINICIAN",
            }
        }
    )
    user_id: str
    role: Role


class LoginRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username": "doctor.jones",
                "password": "strong-password",
                "role": "CLINICIAN",
            }
        }
    )
    username: str
    password: str
    role: Role


class TokenEnvelope(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_expires_in": 900,
                "refresh_expires_in": 604800,
                "user": {"user_id": "doctor.jones", "role": "CLINICIAN"},
            }
        }
    )
    access_token: str = ""
    refresh_token: str = ""
    access_expires_in: int
    refresh_expires_in: int
    user: UserContext


class PatientSummary(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient_id": "P-001",
                "name": "Patient P-001",
                "updated_at": "2026-04-08T09:30:00Z",
            }
        }
    )
    patient_id: str
    name: str
    updated_at: datetime
    probability: float | None = None
    risk_level: str | None = None
    disease_classification: dict[str, Any] | None = None


class PatientListResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"items": [{"patient_id": "P-001", "name": "Patient P-001", "updated_at": "2026-04-08T09:30:00Z"}]}})
    items: list[PatientSummary]


class FeatureVector(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient_id": "P-001",
                "features": {
                    "Age": 73,
                    "MMSE": 24,
                    "FunctionalAssessment": 6.2,
                    "ADL": 6.0,
                    "SleepQuality": 5.1,
                },
            }
        }
    )
    patient_id: str
    features: dict[str, float]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "0a1b2c3d",
                "patient_id": "P-001",
                "queued_phases": [
                    "connectome_inference",
                    "genomic_risk_score",
                    "temporal_forecast",
                    "causal_analysis",
                    "report_generation",
                ],
            }
        }
    )
    job_id: str
    patient_id: str
    queued_phases: list[str]


class ReportRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"patient_id": "P-001", "notes": "Recent decline in sleep quality."}
        }
    )
    patient_id: str
    notes: str | None = None


class ReportResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "task_id": "71ea1452",
                "patient_id": "P-001",
                "status": "queued",
            }
        }
    )
    task_id: str
    patient_id: str
    status: str
    report: dict[str, Any] | None = None


class CausalRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient_id": "P-001",
                "interventions": {"SleepQuality": 7.0},
            }
        }
    )
    patient_id: str
    interventions: dict[str, float] = Field(default_factory=dict)


class CausalResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "task_id": "ccf9930c",
                "patient_id": "P-001",
                "status": "queued",
            }
        }
    )
    task_id: str
    patient_id: str
    status: str


class BiomarkerEvent(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "phase": "temporal_forecast",
                "task_id": "0a1b2c3d",
                "progress": 40,
                "patient_id_hash": "abc123def4567890",
                "timestamp": "2026-04-08T10:00:00Z",
            }
        }
    )
    phase: str
    task_id: str
    progress: int
    patient_id_hash: str | None = None
    timestamp: datetime


class HealthResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"status": "ok"}})
    status: str


class ReadyResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {
            "status": "ready", "database": True, "redis": True,
            "models_loaded": True, "rag_enabled": False,
            "fusion_loaded": False, "pgvector_ok": False,
            "schema_version": "v5",
        }}
    )
    status: str
    database: bool
    redis: bool
    models_loaded: bool
    # v5 additions
    rag_enabled: bool = False
    fusion_loaded: bool = False
    pgvector_ok: bool = False
    schema_version: str = "v5"
    startup_error: str = ""


class QueueDepthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"queue": "celery", "depth": 3}}
    )
    queue: str
    depth: int


class PipelineRunRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"project_name": "neurosynth", "model_name": "phase4-tft"}
        }
    )
    project_name: str = "neurosynth"
    model_name: str = "default"


class PipelineRunResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"run_id": "f2bc2d12-553d-4e3e-90d4-f6d9df66cb56"}}
    )
    run_id: str


class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"error": "rate_limit_exceeded", "detail": "Too many requests"}}
    )
    error: str
    detail: str


class MetricPayload(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"content_type": "text/plain"}})
    content_type: str
    content: str


class SSEHandshakeResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"stream": "/biomarkers/stream"}})
    stream: str


class CeleryTaskResult(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"task_id": "0a1b2c3d", "phase": "connectome_inference", "state": "PENDING", "meta": {}}})
    task_id: str
    phase: str
    state: str
    meta: dict[str, Any] = Field(default_factory=dict)
