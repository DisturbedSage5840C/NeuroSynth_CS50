from __future__ import annotations

from neurosynth.data.contracts import TABLE_SCHEMAS, validate_frame


def test_all_domain_tables_have_contracts() -> None:
    expected = {
        "patients",
        "imaging_sessions",
        "connectivity_matrices",
        "genomic_variants",
        "biomarker_timeseries",
        "clinical_notes",
        "causal_graphs",
        "model_predictions",
    }
    assert expected.issubset(set(TABLE_SCHEMAS))


def test_contract_validation_accepts_synthetic_data(
    synthetic_patients,
    synthetic_imaging_sessions,
    synthetic_connectivity,
    synthetic_variants,
    synthetic_biomarker_timeseries,
    synthetic_notes,
    synthetic_causal_edges,
    synthetic_predictions,
) -> None:
    validate_frame("patients", synthetic_patients)
    validate_frame("imaging_sessions", synthetic_imaging_sessions)
    validate_frame("connectivity_matrices", synthetic_connectivity)
    validate_frame("genomic_variants", synthetic_variants)
    validate_frame("biomarker_timeseries", synthetic_biomarker_timeseries)
    validate_frame("clinical_notes", synthetic_notes)
    validate_frame("causal_graphs", synthetic_causal_edges)
    validate_frame("model_predictions", synthetic_predictions)
