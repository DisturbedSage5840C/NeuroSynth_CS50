from __future__ import annotations

from neurosynth.data.dask_features import build_patient_feature_matrix


def test_build_patient_feature_matrix(
    synthetic_biomarker_timeseries,
    synthetic_connectivity,
    synthetic_variants,
) -> None:
    features = build_patient_feature_matrix(
        biomarker_timeseries=synthetic_biomarker_timeseries,
        connectivity_matrices=synthetic_connectivity,
        genomic_variants=synthetic_variants,
    )

    assert "patient_id" in features.columns
    assert "variant_count" in features.columns
    assert "connectivity_mean" in features.columns
    assert len(features) == 1
