from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest


@pytest.fixture
def synthetic_patients() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": "001_S_0001",
                "patient_cohort": "ADNI",
                "ingestion_date": date(2026, 4, 8),
                "sex": "F",
                "birth_year": 1948,
                "education_years": 16.0,
                "apoe_e4_count": 1,
            }
        ]
    )


@pytest.fixture
def synthetic_imaging_sessions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "imaging_session_id": "sess-1",
                "patient_id": "001_S_0001",
                "patient_cohort": "ADNI",
                "ingestion_date": date(2026, 4, 8),
                "series_uid": "1.2.3",
                "modality": "MR",
                "field_strength_t": 3.0,
                "voxel_size_mm": "(1.0,1.0,1.0)",
                "orientation": "100010",
                "qc_pass": True,
                "qc_flags": [],
                "registered_nifti_uri": "s3://warehouse/sess-1.nii.gz",
            }
        ]
    )


@pytest.fixture
def synthetic_connectivity() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "connectivity_id": "conn-1",
                "imaging_session_id": "sess-1",
                "patient_id": "001_S_0001",
                "patient_cohort": "ADNI",
                "ingestion_date": date(2026, 4, 8),
                "atlas_name": "aal",
                "n_regions": 2,
                "matrix_uri": "s3://warehouse/conn-1.parquet",
                "mean_connectivity": 0.34,
            }
        ]
    )


@pytest.fixture
def synthetic_variants() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant_id": "var-1",
                "patient_id": "001_S_0001",
                "patient_cohort": "ADNI",
                "ingestion_date": date(2026, 4, 8),
                "chrom": "chr19",
                "pos": 45411941,
                "ref": "C",
                "alt": "T",
                "gene": "APOE",
                "clinvar_significance": "Likely_pathogenic",
                "dbsnp_id": "rs429358",
            }
        ]
    )


@pytest.fixture
def synthetic_biomarker_timeseries() -> pd.DataFrame:
    t0 = datetime(2026, 4, 8, 10, 0, tzinfo=timezone.utc)
    return pd.DataFrame(
        [
            {
                "timeseries_id": "ts-1",
                "patient_id": "001_S_0001",
                "patient_cohort": "ADNI",
                "ingestion_date": date(2026, 4, 8),
                "modality": "heart_rate",
                "window_start": t0,
                "window_end": t0,
                "metric_mean": 78.0,
                "metric_std": 1.2,
                "sample_count": 10,
            },
            {
                "timeseries_id": "ts-2",
                "patient_id": "001_S_0001",
                "patient_cohort": "ADNI",
                "ingestion_date": date(2026, 4, 8),
                "modality": "eeg",
                "window_start": t0,
                "window_end": t0,
                "metric_mean": 0.45,
                "metric_std": 0.05,
                "sample_count": 10,
            },
        ]
    )


@pytest.fixture
def synthetic_notes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "note_id": "note-1",
                "patient_id": "001_S_0001",
                "patient_cohort": "ADNI",
                "ingestion_date": date(2026, 4, 8),
                "encounter_time": datetime(2026, 4, 8, 11, 0, tzinfo=timezone.utc),
                "note_text": "Mild memory complaint.",
                "source_system": "ehr",
            }
        ]
    )


@pytest.fixture
def synthetic_causal_edges() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "graph_id": "g-1",
                "patient_id": "001_S_0001",
                "patient_cohort": "ADNI",
                "ingestion_date": date(2026, 4, 8),
                "source_node": "SleepQuality",
                "target_node": "Diagnosis",
                "edge_weight": 0.52,
                "edge_type": "direct",
            }
        ]
    )


@pytest.fixture
def synthetic_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "prediction_id": "pred-1",
                "patient_id": "001_S_0001",
                "patient_cohort": "ADNI",
                "ingestion_date": date(2026, 4, 8),
                "model_name": "neurosynth_ensemble",
                "model_version": "2.0.0",
                "prediction_time": datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc),
                "risk_score": 0.27,
                "risk_label": "Low",
            }
        ]
    )
