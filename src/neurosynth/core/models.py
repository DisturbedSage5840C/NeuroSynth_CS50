from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

class BiomarkerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: UUID
    visit_code: str
    collection_date: date
    abeta42_pgml: float | None = None
    ptau181_pgml: float | None = None
    total_tau_pgml: float | None = None
    nfl_pgml: float | None = None
    alpha_syn_pgml: float | None = None
    hippocampal_volume_mm3: float | None = None
    ventricle_volume_mm3: float | None = None
    cdrsb_score: float | None = None
    mmse_score: float | None = None
    moca_score: float | None = None
    updrs_part3: float | None = None
    site_id: str | None = None
    harmonized_flag: bool = False
    feature_vector: list[float] = Field(default_factory=list)
    embedding_model_version: str | None = None


class DicomValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    modality: str | None = None
    manufacturer: str | None = None
    field_strength: float | None = None
    series_description: str | None = None
    pixel_spacing: list[float] = Field(default_factory=list)
    slice_thickness: float | None = None
    n_slices: int | None = None
    has_private_tags: bool = False


class DicomResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: Path
    patient_uuid: UUID
    validation: DicomValidationResult
    s3_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class WearableRawSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: UUID
    device_id: str
    timestamp_ns: int
    x_g: float
    y_g: float
    z_g: float
    temperature_c: float
    light_lux: float


class ComputedFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: UUID
    window_start: datetime
    window_duration_secs: int
    gait_cadence: float
    tremor_index: float
    bradykinesia_score: float
    freezing_episodes: int
    step_count: int
    sleep_duration_mins: int


class OAuthToken(BaseModel):
    model_config = ConfigDict(extra="forbid")

    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    acquired_at: datetime
