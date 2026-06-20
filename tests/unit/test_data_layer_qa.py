from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from neurosynth.core.config import NeuroSynthSettings
from neurosynth.data.contracts import validate_frame
from neurosynth.data.genomics_pipeline import GenomicsIngestionPipeline
from neurosynth.dicom.processor import DICOMProcessor


class _FakeVariantRecord:
    def __init__(self, chrom: str, pos: int, ref: str, alt: str, gene: str) -> None:
        self.chrom = chrom
        self.pos = pos
        self.ref = ref
        self.alts = (alt,)
        self.info = {"GENE": gene}


class _FakeVariantFile:
    def __init__(self, records: list[_FakeVariantRecord]) -> None:
        self._records = records

    def __enter__(self) -> "_FakeVariantFile":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _ = (exc_type, exc, tb)

    def fetch(self):
        for r in self._records:
            yield r


def test_dicom_parser_synthetic_3x3x3_volume_has_expected_metadata(synthetic_dicom_file: Path) -> None:
    processor = DICOMProcessor(NeuroSynthSettings())

    result = processor.validate_dicom(synthetic_dicom_file)

    assert result.is_valid is True
    assert result.modality == "MR"
    assert result.n_slices == 3
    assert result.pixel_spacing == [1.0, 1.0]
    assert result.slice_thickness == 1.0


def test_genomic_vcf_parser_with_synthetic_fixture_and_contract_validation(
    monkeypatch: pytest.MonkeyPatch,
    fake_iceberg: object,
    tmp_path: Path,
    synthetic_vcf_text: str,
) -> None:
    vcf_path = tmp_path / "synthetic.vcf"
    h5_path = tmp_path / "synthetic.h5"
    vcf_path.write_text(synthetic_vcf_text, encoding="utf-8")

    records = [
        _FakeVariantRecord("1", 1000, "A", "G", "APOE"),
        _FakeVariantRecord("1", 1100, "C", "T", "TREM2"),
    ]

    monkeypatch.setattr("neurosynth.data.genomics_pipeline.pysam.VariantFile", lambda *_args, **_kwargs: _FakeVariantFile(records))
    monkeypatch.setattr(
        "neurosynth.data.genomics_pipeline.allel.read_vcf",
        lambda *_args, **_kwargs: {
            "variants/CHROM": np.array(["1", "1"]),
            "variants/POS": np.array([1000, 1100]),
            "variants/REF": np.array(["A", "C"]),
            "variants/ALT": np.array(["G", "T"]),
        },
    )

    # Avoid external chain files in unit tests.
    pipeline = GenomicsIngestionPipeline(iceberg=fake_iceberg)
    pipeline.liftover = SimpleNamespace(convert_coordinate=lambda chrom, pos: [(chrom, pos, "+", 0)])

    frame = pipeline.ingest_vcf(
        vcf_path=vcf_path,
        patient_id="P-UNIT-001",
        patient_cohort="TEST",
        h5_out_path=h5_path,
    )

    validated = validate_frame("genomic_variants", frame)
    assert len(validated) == 2
    assert h5_path.exists()
    assert "genomic_variants" in fake_iceberg.records


def test_iceberg_like_in_memory_pyarrow_roundtrip_is_schema_stable() -> None:
    frame = pd.DataFrame(
        {
            "variant_id": ["v1"],
            "patient_id": ["P-UNIT-001"],
            "patient_cohort": ["TEST"],
            "ingestion_date": [pd.Timestamp("2026-04-08").date()],
            "chrom": ["1"],
            "pos": [1234],
            "ref": ["A"],
            "alt": ["G"],
            "gene": ["APOE"],
            "clinvar_significance": ["pathogenic"],
            "dbsnp_id": ["rs123"],
        }
    )
    validated = validate_frame("genomic_variants", frame)

    arrow_table = pa.Table.from_pandas(validated, preserve_index=False)
    restored = arrow_table.to_pandas()

    assert len(restored) == 1
    assert restored.loc[0, "gene"] == "APOE"
