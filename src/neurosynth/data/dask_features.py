from __future__ import annotations

import dask.dataframe as dd
import pandas as pd

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

def build_patient_feature_matrix(
    biomarker_timeseries: pd.DataFrame,
    connectivity_matrices: pd.DataFrame,
    genomic_variants: pd.DataFrame,
) -> pd.DataFrame:
    """Build a patient-level feature matrix with Dask aggregation pipelines."""

    ts_dd = dd.from_pandas(biomarker_timeseries, npartitions=max(1, len(biomarker_timeseries) // 1000 or 1))
    conn_dd = dd.from_pandas(connectivity_matrices, npartitions=max(1, len(connectivity_matrices) // 1000 or 1))
    var_dd = dd.from_pandas(genomic_variants, npartitions=max(1, len(genomic_variants) // 1000 or 1))

    ts_grouped = ts_dd.groupby(["patient_id", "modality"])["metric_mean"].mean().reset_index().compute()
    ts_agg = ts_grouped.pivot_table(index="patient_id", columns="modality", values="metric_mean", aggfunc="mean").reset_index()

    conn_agg = conn_dd.groupby("patient_id")["mean_connectivity"].mean().reset_index().compute().rename(columns={"mean_connectivity": "connectivity_mean"})
    var_agg = var_dd.groupby("patient_id").size().reset_index().compute().rename(columns={0: "variant_count"})

    result = ts_agg.merge(conn_agg, on="patient_id", how="outer").merge(var_agg, on="patient_id", how="outer")

    for col in result.columns:
        if col != "patient_id":
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)

    return result
