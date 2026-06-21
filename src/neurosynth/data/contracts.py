# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

from datetime import date

import pandas as pd
import pandera as pa
from pandera import Check


def _ts_col(nullable: bool = False) -> pa.Column:
    return pa.Column(pa.DateTime, nullable=nullable, coerce=True)


PATIENTS_SCHEMA = pa.DataFrameSchema(
    {
        "patient_id": pa.Column(str, nullable=False),
        "patient_cohort": pa.Column(str, nullable=False),
        "ingestion_date": pa.Column(date, nullable=False, coerce=True),
        "sex": pa.Column(str, nullable=True),
        "birth_year": pa.Column(int, Check.in_range(1900, 2100), nullable=True, coerce=True),
        "education_years": pa.Column(float, Check.in_range(0, 30), nullable=True, coerce=True),
        "apoe_e4_count": pa.Column(int, Check.in_range(0, 2), nullable=True, coerce=True),
    },
    strict=False,
)

IMAGING_SESSIONS_SCHEMA = pa.DataFrameSchema(
    {
        "imaging_session_id": pa.Column(str, nullable=False),
        "patient_id": pa.Column(str, nullable=False),
        "patient_cohort": pa.Column(str, nullable=False),
        "ingestion_date": pa.Column(date, nullable=False, coerce=True),
        "series_uid": pa.Column(str, nullable=False),
        "modality": pa.Column(str, nullable=False),
        "field_strength_t": pa.Column(float, Check.ge(0.0), nullable=True, coerce=True),
        "voxel_size_mm": pa.Column(object, nullable=True),
        "orientation": pa.Column(str, nullable=True),
        "qc_pass": pa.Column(bool, nullable=False),
        "qc_flags": pa.Column(object, nullable=True),
        "registered_nifti_uri": pa.Column(str, nullable=True),
    },
    strict=False,
)

CONNECTIVITY_MATRICES_SCHEMA = pa.DataFrameSchema(
    {
        "connectivity_id": pa.Column(str, nullable=False),
        "imaging_session_id": pa.Column(str, nullable=False),
        "patient_id": pa.Column(str, nullable=False),
        "patient_cohort": pa.Column(str, nullable=False),
        "ingestion_date": pa.Column(date, nullable=False, coerce=True),
        "atlas_name": pa.Column(str, nullable=False),
        "n_regions": pa.Column(int, Check.gt(0), nullable=False, coerce=True),
        "matrix_uri": pa.Column(str, nullable=False),
        "mean_connectivity": pa.Column(float, nullable=True, coerce=True),
    },
    strict=False,
)

GENOMIC_VARIANTS_SCHEMA = pa.DataFrameSchema(
    {
        "variant_id": pa.Column(str, nullable=False),
        "patient_id": pa.Column(str, nullable=False),
        "patient_cohort": pa.Column(str, nullable=False),
        "ingestion_date": pa.Column(date, nullable=False, coerce=True),
        "chrom": pa.Column(str, nullable=False),
        "pos": pa.Column(int, Check.gt(0), nullable=False, coerce=True),
        "ref": pa.Column(str, nullable=False),
        "alt": pa.Column(str, nullable=False),
        "gene": pa.Column(str, nullable=True),
        "clinvar_significance": pa.Column(str, nullable=True),
        "dbsnp_id": pa.Column(str, nullable=True),
    },
    strict=False,
)

BIOMARKER_TIMESERIES_SCHEMA = pa.DataFrameSchema(
    {
        "timeseries_id": pa.Column(str, nullable=False),
        "patient_id": pa.Column(str, nullable=False),
        "patient_cohort": pa.Column(str, nullable=False),
        "ingestion_date": pa.Column(date, nullable=False, coerce=True),
        "modality": pa.Column(str, nullable=False),
        "window_start": _ts_col(),
        "window_end": _ts_col(),
        "metric_mean": pa.Column(float, nullable=True, coerce=True),
        "metric_std": pa.Column(float, nullable=True, coerce=True),
        "sample_count": pa.Column(int, Check.ge(1), nullable=False, coerce=True),
    },
    strict=False,
)

CLINICAL_NOTES_SCHEMA = pa.DataFrameSchema(
    {
        "note_id": pa.Column(str, nullable=False),
        "patient_id": pa.Column(str, nullable=False),
        "patient_cohort": pa.Column(str, nullable=False),
        "ingestion_date": pa.Column(date, nullable=False, coerce=True),
        "encounter_time": _ts_col(nullable=True),
        "note_text": pa.Column(str, nullable=False),
        "source_system": pa.Column(str, nullable=True),
    },
    strict=False,
)

CAUSAL_GRAPHS_SCHEMA = pa.DataFrameSchema(
    {
        "graph_id": pa.Column(str, nullable=False),
        "patient_id": pa.Column(str, nullable=True),
        "patient_cohort": pa.Column(str, nullable=False),
        "ingestion_date": pa.Column(date, nullable=False, coerce=True),
        "source_node": pa.Column(str, nullable=False),
        "target_node": pa.Column(str, nullable=False),
        "edge_weight": pa.Column(float, nullable=False, coerce=True),
        "edge_type": pa.Column(str, nullable=False),
    },
    strict=False,
)

MODEL_PREDICTIONS_SCHEMA = pa.DataFrameSchema(
    {
        "prediction_id": pa.Column(str, nullable=False),
        "patient_id": pa.Column(str, nullable=False),
        "patient_cohort": pa.Column(str, nullable=False),
        "ingestion_date": pa.Column(date, nullable=False, coerce=True),
        "model_name": pa.Column(str, nullable=False),
        "model_version": pa.Column(str, nullable=False),
        "prediction_time": _ts_col(),
        "risk_score": pa.Column(float, Check.in_range(0.0, 1.0), nullable=False, coerce=True),
        "risk_label": pa.Column(str, nullable=False),
    },
    strict=False,
)

TABLE_SCHEMAS: dict[str, pa.DataFrameSchema] = {
    "patients": PATIENTS_SCHEMA,
    "imaging_sessions": IMAGING_SESSIONS_SCHEMA,
    "connectivity_matrices": CONNECTIVITY_MATRICES_SCHEMA,
    "genomic_variants": GENOMIC_VARIANTS_SCHEMA,
    "biomarker_timeseries": BIOMARKER_TIMESERIES_SCHEMA,
    "clinical_notes": CLINICAL_NOTES_SCHEMA,
    "causal_graphs": CAUSAL_GRAPHS_SCHEMA,
    "model_predictions": MODEL_PREDICTIONS_SCHEMA,
}


def validate_frame(table_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    if table_name not in TABLE_SCHEMAS:
        raise KeyError(f"Unknown table schema: {table_name}")
    return TABLE_SCHEMAS[table_name].validate(frame, lazy=True)
