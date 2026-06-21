# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""Priority 5 verification — Inference API Refactor.

Wrapped as pytest functions: the LIME perturbation search, counterfactual
generation, and full RandomForest fit at the original module level made
pytest --cov hang for the entire 6h job timeout on CI.
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def test_v2_pydantic_models() -> None:
    from backend.models_v2 import (
        AnalyzeResponseV2,
        CausalIntervention,
        ConfidenceInterval,
        Counterfactual,
        LIMEExplanation,
        RFC7807Error,
        SHAPValue,
        TrajectoryForecast,
    )

    response = AnalyzeResponseV2(
        patient_id="P-001",
        request_id="test-123",
        prediction=0,
        probability=0.35,
        risk_level="Low",
        confidence="Medium",
        shap_values=[SHAPValue(feature="MMSE", value=-0.15)],
        lime_explanation=[LIMEExplanation(feature="MMSE", weight=-0.12, direction="decreases_risk")],
        counterfactuals=[
            Counterfactual(feature="SleepQuality", current_value=4.0, target_value=7.0, risk_delta=-0.08)
        ],
        trajectory_48mo=TrajectoryForecast(months=[6, 12, 18, 24, 30, 36, 42, 48], values=[0.3] * 8),
        causal_interventions=[
            CausalIntervention(factor="PhysicalActivity", effect_size=0.05, direction="protective")
        ],
        confidence_intervals=ConfidenceInterval(method="conformal", coverage=0.95, lower=0.28, upper=0.42),
    )
    assert response.probability == 0.35
    assert len(response.shap_values) == 1
    assert len(response.lime_explanation) == 1
    assert len(response.counterfactuals) == 1
    assert response.trajectory_48mo.months[-1] == 48
    assert response.api_version == "v2"

    error = RFC7807Error(
        type="https://neurosynth.dev/errors/validation",
        title="Validation Error",
        status=422,
        detail="MMSE must be between 0 and 30",
        instance="/v2/predictions/analyze",
        trace_id="test-trace",
    )
    assert error.status == 422


@pytest.fixture(scope="module")
def fitted_rf_and_data():
    """Train the RF once per module so LIME / counterfactual tests share it."""
    from sklearn.ensemble import RandomForestClassifier

    from backend.data_pipeline import DataPipeline

    pipeline = DataPipeline()
    X_train, X_test, y_train, y_test, feature_names, scaler, stats = pipeline.process()
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train.values, y_train.values)

    def predict_proba_fn(X: np.ndarray) -> np.ndarray:
        return rf.predict_proba(X)[:, 1]

    return rf, X_test, feature_names, predict_proba_fn


@pytest.mark.timeout(300)
def test_lime_local_explanations(fitted_rf_and_data) -> None:
    from backend.routers.predictions_v2 import _compute_lime

    _, X_test, feature_names, predict_proba_fn = fitted_rf_and_data
    lime_results = _compute_lime(predict_proba_fn, X_test.values[0], feature_names)
    assert len(lime_results) > 0
    assert all("feature" in r and "weight" in r for r in lime_results)


@pytest.mark.timeout(300)
def test_counterfactual_recommendations(fitted_rf_and_data) -> None:
    from backend.routers.predictions_v2 import _generate_counterfactuals

    rf, X_test, feature_names, predict_proba_fn = fitted_rf_and_data
    current_prob = float(rf.predict_proba(X_test.values[:1])[:, 1][0])
    counterfactuals = _generate_counterfactuals(
        predict_proba_fn, X_test.values[0], feature_names, current_prob,
    )
    assert all("feature" in cf and "risk_delta" in cf for cf in counterfactuals)


def test_circuit_breaker() -> None:
    from backend.routers.predictions_v2 import _CircuitBreaker

    breaker = _CircuitBreaker(threshold=3, reset_timeout=1.0)
    assert not breaker.is_open

    breaker.record_failure()
    breaker.record_failure()
    assert not breaker.is_open
    breaker.record_failure()
    assert breaker.is_open

    breaker.record_success()
    assert not breaker.is_open


def test_v2_router_registration() -> None:
    from backend.routers.predictions_v2 import router as v2_router

    routes = [r.path for r in v2_router.routes]
    assert "/analyze" in routes or any("/analyze" in r for r in routes)
    assert any("health" in r for r in routes)
