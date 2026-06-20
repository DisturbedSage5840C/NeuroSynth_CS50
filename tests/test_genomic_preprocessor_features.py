from __future__ import annotations

import pandas as pd

from neurosynth.genomic.preprocessor import GenomicPreprocessor


def test_variant_feature_matrix_has_16_columns() -> None:
    pre = GenomicPreprocessor()
    df = pd.DataFrame(
        {
            "Uploaded_variation": ["1:100:A:T"],
            "Location": ["1:100"],
            "Allele": ["T"],
            "Consequence": ["missense_variant"],
            "IMPACT": ["MODERATE"],
            "SYMBOL": ["APOE"],
            "CADD_PHRED": [25],
            "gnomADg_AF": [0.001],
            "ClinVar_CLNSIG": ["pathogenic"],
            "SIFT": [0.02],
            "PolyPhen": [0.85],
        }
    )
    mat = pre.build_variant_feature_matrix(
        patient_id="p1",
        annotated_variants=df,
        prs_row={"prs_ad": 0.1, "prs_pd": 0.2, "prs_als": -0.1},
    )
    assert mat.matrix.shape[1] == 16
