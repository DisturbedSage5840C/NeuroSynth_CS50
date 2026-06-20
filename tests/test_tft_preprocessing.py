from __future__ import annotations

from pathlib import Path

import pandas as pd

from neurosynth.temporal_tft.preprocessing import BiomarkerTimeSeriesPreprocessor


def test_build_longitudinal_dataset(tmp_path: Path) -> None:
    adni = pd.DataFrame(
        {
            "PTID": ["p1", "p1", "p2"],
            "VISCODE": ["BL", "M06", "BL"],
            "RID": [1, 1, 2],
            "AGE": [70, 70.5, 68],
            "PTGENDER": ["M", "M", "F"],
            "PTEDUCAT": [16, 16, 14],
            "DX_bl": ["CN", "CN", "CN"],
            "CDRSB": [0.1, 0.2, 0.0],
            "Hippocampus": [4000, 3950, 4200],
            "ADAS13": [10, 11, 9],
            "nfl_pgml": [12, 13, 11],
            "visit_date": ["2020-01-01", "2020-07-01", "2020-01-01"],
            "patient_id": ["p1", "p1", "p2"],
        }
    )
    ppmi = pd.DataFrame(
        {
            "patient_id": ["p3"],
            "month": [0],
            "CDRSB": [0.3],
            "Hippocampus": [3900],
            "ADAS13": [12],
            "nfl_pgml": [15],
            "visit_date": ["2020-01-01"],
        }
    )

    adni_path = tmp_path / "adni.csv"
    ppmi_path = tmp_path / "ppmi.csv"
    out_path = tmp_path / "longitudinal.parquet"
    adni.to_csv(adni_path, index=False)
    ppmi.to_csv(ppmi_path, index=False)

    pre = BiomarkerTimeSeriesPreprocessor()
    df = pre.build_longitudinal_dataset(adni_path, ppmi_path, out_path)

    assert "dci" in df.columns
    assert "time_idx" in df.columns
    assert out_path.exists()
