from __future__ import annotations

import numpy as np

from neurosynth.genomics.phase3_variant_transformer import GenomicPhase3Model, Phase3Config


def test_phase3_predict_with_uncertainty_smoke() -> None:
    model = GenomicPhase3Model(Phase3Config(mc_samples=8))

    x = np.array(
        [
            [0.1, 0.2, 0.0],
            [0.2, 0.3, 1.0],
            [0.3, 0.1, 0.0],
            [0.7, 0.8, 1.0],
        ],
        dtype=float,
    )
    model.fit(
        variant_features=x,
        risk_target=np.array([0.6], dtype=float),
        pathogenic_target=np.array([1.0], dtype=float),
        use_dp=False,
    )
    out = model.predict_with_uncertainty(x)

    assert "mean" in out
    assert "lower_80" in out
    assert "upper_95" in out
    assert len(out["shap_values"]) > 0
