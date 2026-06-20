from __future__ import annotations

import pandas as pd

from neurosynth.causal.data_prep import CausalDataPreparer


def test_prepare_causal_matrix_shapes() -> None:
    rows = []
    for pid in ["p1", "p2"]:
        for t in range(5):
            rows.append(
                {
                    "patient_id": pid,
                    "time_idx": t,
                    "abeta42": 900 + t,
                    "ptau181": 30 + t,
                    "total_tau": 300 + t,
                    "alpha_syn": 2000 + t,
                    "nfl": 12 + t,
                    "hippocampus": 3800 - t * 5,
                    "entorhinal": 2200 - t * 3,
                    "fusiform": 1800 - t * 2,
                    "midtemp": 1700 - t * 2,
                    "ventricles": 25000 + t * 100,
                    "wholebrain": 1100000 - t * 500,
                    "cdrsb": 1 + 0.1 * t,
                    "mmse": 28 - 0.1 * t,
                    "moca": 25 - 0.1 * t,
                    "adas13": 10 + 0.5 * t,
                    "updrs3": 8 + t,
                    "gait_speed": 1.1 - 0.01 * t,
                    "sleep_efficiency": 0.8 - 0.01 * t,
                    "step_count": 6500 - 30 * t,
                    "tremor_index": 0.2 + 0.02 * t,
                    "bradykinesia_score": 0.3 + 0.03 * t,
                    "age": 72,
                    "sex_male": 1,
                    "education_years": 16,
                    "apoe_e4_count": 1,
                    "prs_ad": 0.4,
                    "inflammation_proxy": 0.2,
                    "dci": 15 + t,
                }
            )
    df = pd.DataFrame(rows)
    prep = CausalDataPreparer()
    ci = prep.prepare_causal_matrix(df)

    assert ci.patient_matrix.shape[1] == 28
    assert ci.patient_delta_matrix.shape[1] == 56
    assert ci.population_matrix.shape[1] == 28
