from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neurosynth.connectors.adni import ADNIConnector
from neurosynth.core.config import NeuroSynthSettings


@pytest.mark.asyncio
async def test_adni_load_sanitizes_negative_values(tmp_path: Path) -> None:
    adni = pd.DataFrame(
        {
            "PTID": ["001_S_0001"],
            "VISCODE": ["M06"],
            "CDRSB": [-1.0],
            "ADAS13": [12.0],
            "MMSE": [28.0],
            "Ventricles": [2.0],
            "Hippocampus": [3.0],
            "WholeBrain": [4.0],
            "Entorhinal": [5.0],
            "Fusiform": [6.0],
            "MidTemp": [7.0],
            "ICV": [8.0],
            "ABETA": [9.0],
            "TAU": [10.0],
            "PTAU": [11.0],
            "SITE": ["A"],
        }
    )
    upenn = pd.DataFrame({"PTID": ["001_S_0001"], "VISCODE": ["M06"]})

    adni_path = tmp_path / "ADNIMERGE.csv"
    upenn_path = tmp_path / "UPENNBIOMK.csv"
    adni.to_csv(adni_path, index=False)
    upenn.to_csv(upenn_path, index=False)

    connector = ADNIConnector(NeuroSynthSettings(adni_sftp_host="example.org"))
    await connector.load_files(str(adni_path), str(upenn_path))
    await connector.validate_schema()

    row = (await connector.fetch_batch(0, 1))[0]
    assert pd.isna(row["CDRSB"])
    assert row["harmonized_flag"] is True
