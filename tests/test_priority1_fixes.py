"""Priority 1 bug-fix verification — wrapped as a pytest test so the heavy work
(disease classifier training, report generation) runs only when pytest invokes
the test, not at module collection time. Module-level execution previously
hung pytest --cov for hours on CI.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pytest

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@pytest.mark.timeout(300)
def test_priority1_all_fixes() -> None:
    # 1. Disease classifier — real-data training
    from backend.disease_classifier import DiseaseClassifier

    clf = DiseaseClassifier()
    clf.train()
    r = clf.predict_disease({"Age": 73, "MMSE": 18, "MemoryComplaints": 1})
    assert "predicted_disease" in r and "confidence" in r
    assert clf.feature_names is not None and len(clf.feature_names) > 14, (
        "Feature alignment fix failed"
    )

    # 2. Causal engine — safe variable lookups
    from backend.causal_engine import NeuralCausalDiscovery

    causal = NeuralCausalDiscovery(variables=["A", "B", "C"])
    causal.latest_W = np.zeros((3, 3))
    graph = causal.get_causal_graph()
    assert "top_causes_of_Diagnosis" in graph and graph["top_causes_of_Diagnosis"] == []
    assert "top_causes_of_MMSE" in graph and graph["top_causes_of_MMSE"] == []

    # 3. Report generator — sync httpx
    from backend.report_generator import ClinicalReportGenerator

    rg = ClinicalReportGenerator(hf_token=None)
    report = rg.generate_report(
        patient_data={"Age": 73, "MMSE": 18},
        prediction={"probability": 0.7, "risk_level": "High", "confidence": "High"},
        trajectory=[0.7, 0.72, 0.74],
        causal_graph={},
        shap_values=[{"feature": "MMSE", "value": -0.3}],
    )
    assert "sections" in report

    # 4. Biomarker model — file-not-found handling
    from backend.biomarker_model import BiomarkerPredictor

    bp = BiomarkerPredictor(
        feature_names=["Age", "MMSE"], models_dir="/tmp/nonexistent_neurosynth_test"
    )
    with pytest.raises(Exception):
        bp.load_from_disk()

    # 5. Config — prod secret validation
    prev_env = os.environ.get("NEUROSYNTH_APP_ENV")
    prev_jwt = os.environ.get("NEUROSYNTH_JWT_SECRET")
    os.environ["NEUROSYNTH_APP_ENV"] = "prod"
    os.environ["NEUROSYNTH_JWT_SECRET"] = "change-me"
    try:
        from backend.core.config import Settings

        with pytest.raises(ValueError):
            Settings()
    finally:
        if prev_env is None:
            os.environ.pop("NEUROSYNTH_APP_ENV", None)
        else:
            os.environ["NEUROSYNTH_APP_ENV"] = prev_env
        if prev_jwt is None:
            os.environ.pop("NEUROSYNTH_JWT_SECRET", None)
        else:
            os.environ["NEUROSYNTH_JWT_SECRET"] = prev_jwt
