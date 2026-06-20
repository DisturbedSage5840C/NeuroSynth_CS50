from __future__ import annotations

import numpy as np
import pandas as pd

from neurosynth.temporal_tft.dataset_factory import DatasetFactory


def _synthetic_df() -> pd.DataFrame:
    rows = []
    for pid in ["p1", "p2", "p3"]:
        for t in range(10):
            rows.append(
                {
                    "patient_id": pid,
                    "time_idx": t,
                    "visit_date": pd.Timestamp("2020-01-01") + pd.DateOffset(months=6 * t),
                    "dci": float(t + np.random.rand()),
                    "sex": "M",
                    "apoe_e4_cat": "1",
                    "disease_subtype": "AD",
                    "cohort": "ADNI",
                    "site_region": "NA",
                    "age_at_enrollment": 70.0,
                    "education_years": 16.0,
                    "prs_ad_normalized": 0.2,
                    "prs_pd_normalized": 0.1,
                    "apoe_e4_count": 1.0,
                    "medication_class": "none",
                    "season": "winter",
                    "age_at_visit": 70 + t * 0.5,
                    "visit_number": t + 1,
                    "months_since_diagnosis": t * 6,
                    "total_drug_burden_score": 0.0,
                    "csf_abeta42": 1000.0,
                    "csf_ptau181": 20.0,
                    "csf_total_tau": 200.0,
                    "csf_ratio": 40.0,
                    "hippocampal_volume": 3800.0,
                    "entorhinal_volume": 2200.0,
                    "fusiform_volume": 1800.0,
                    "ventricle_volume": 25000.0,
                    "whole_brain_volume": 1100000.0,
                    "atrophy_asymmetry": 0.05,
                    "cdrsb": 1.0,
                    "mmse": 28.0,
                    "moca": 25.0,
                    "adas13": 12.0,
                    "nfl_plasma": 15.0,
                    "alpha_syn_csf": 2.0,
                    "gait_speed": 1.1,
                    "tremor_index": 0.2,
                    "bradykinesia_score": 0.3,
                    "step_count_daily": 6000.0,
                    "sleep_efficiency": 0.8,
                    "delta_hippocampus": -10.0,
                    "delta_nfl": 0.5,
                    "delta_cdrsb": 0.1,
                    "accel_hippocampus": -1.0,
                    "accel_nfl": 0.05,
                }
            )
    return pd.DataFrame(rows)


def test_dataset_factory_and_sampler() -> None:
    df = _synthetic_df()
    factory = DatasetFactory()
    train_ds, val_ds, test_ds = factory.create_datasets(df, "2022-01-01", "2023-01-01")
    sampler = factory.create_weighted_sampler(train_ds)

    assert len(train_ds) > 0
    assert len(val_ds) >= 0
    assert len(test_ds) >= 0
    assert sampler.num_samples > 0
