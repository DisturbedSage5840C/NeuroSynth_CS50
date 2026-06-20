from __future__ import annotations

import os
from pathlib import Path

import pytest

from neurosynth.genomic.preprocessor import GenomicPreprocessor


@pytest.mark.integration
def test_1000g_reference_qc_smoke() -> None:
    vcf = os.environ.get("THOUSAND_GENOMES_VCF")
    if not vcf:
        pytest.skip("Set THOUSAND_GENOMES_VCF to run integration smoke test")

    pre = GenomicPreprocessor()
    out_prefix = Path("/tmp/neurosynth_1000g_qc")
    report = pre.run_full_qc_pipeline(Path(vcf), out_prefix)
    assert report.output_prefix == out_prefix
