from __future__ import annotations

import pandas as pd

from neurosynth.genomic.risk import VariantRiskScorer


def test_cadd_burden_nonzero() -> None:
    df = pd.DataFrame(
        {
            "gene_symbol": ["APOE", "APOE", "APP"],
            "impact_score": [2, 1, 0],
            "gnomad_af": [0.001, 0.005, 0.2],
            "cadd_phred": [20.0, 10.0, 5.0],
        }
    )
    scorer = VariantRiskScorer()
    burden = scorer.compute_cadd_burden(df, "APOE")
    assert burden > 0
