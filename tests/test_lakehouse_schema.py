from __future__ import annotations

from neurosynth.core.config import NeuroSynthSettings
from neurosynth.lakehouse.iceberg import NeuroSynthLakehouse


def test_lakehouse_has_required_tables() -> None:
    lakehouse = NeuroSynthLakehouse(NeuroSynthSettings())
    names = {table.name for table in lakehouse._table_defs()}
    assert "neurosynth.patients" in names
    assert "neurosynth.biomarker_longitudinal" in names
    assert "neurosynth.imaging_index" in names
    assert "neurosynth.genomics" in names
