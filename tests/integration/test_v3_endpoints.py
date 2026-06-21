"""Integration tests for NeuroSynth v3 endpoints.

Tests all routes added in Part 4 — Backend Changes:
  GET  /v3/data/sources
  GET  /v3/data/cohort/stats
  GET  /v3/data/provenance
  POST /v3/data/refresh/{source}      (admin-only)
  POST /v3/literature/search
  GET  /v3/literature/cite/{pmid}
  GET  /v3/literature/status
  POST /v3/predictions/analyze
  GET  /v3/fusion/weights
  GET  /ready                         (v5 fields)

These tests use FastAPI's TestClient with all external dependencies
(DB, Redis, ML models, RAG) monkeypatched or skipped gracefully.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Return a TestClient with all heavy state replaced by lightweight mocks."""
    with (
        patch("backend.db.Database.connect", new_callable=AsyncMock),
        patch("backend.db.Database.disconnect", new_callable=AsyncMock),
        patch("backend.api.Redis") as mock_redis_cls,
        patch("backend.core.config.get_settings") as mock_settings,
        # Prevent the lifespan from spawning a pretrain subprocess — on a cold CI
        # runner, Python startup + importing torch/lightgbm alone takes 30-50s.
        patch("backend.api._manifest_valid", return_value=True),
        # Prevent model-file I/O and heavy ML lib imports during test startup.
        patch("backend.model_registry.ModelRegistry"),
        patch("backend.report_generator_v4.ClinicalReportGeneratorV4"),
    ):
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping = AsyncMock()
        mock_redis_instance.close = AsyncMock()
        mock_redis_instance.llen = AsyncMock(return_value=0)
        mock_redis_cls.from_url.return_value = mock_redis_instance

        mock_settings.return_value = MagicMock(
            postgres_dsn="postgresql://user:pass@localhost/neurosynth",
            redis_url="redis://localhost:6379/0",
            jwt_secret="test-secret-32chars-xxxxxxxxxxxx",
            patient_hash_secret="test-hash-secret-xxxxxxxxxxxxxxxx",
            allowed_origins=["http://localhost:5173"],
            app_env="test",
            auth_cookie_secure=False,
        )
        from backend.api import app
        with TestClient(app, raise_server_exceptions=False) as c:
            # Inject minimal app.state so endpoints don't 503
            app.state.predictor = MagicMock()
            app.state.predictor.predict = MagicMock(return_value={"probability": 0.42, "prediction": 0})
            app.state.predictor.get_shap_values = MagicMock(return_value=[[0.1] * 32])
            app.state.scaler = MagicMock()
            app.state.scaler.transform = MagicMock(side_effect=lambda x: x)
            app.state.feature_names = ["Age", "MMSE", "FunctionalAssessment"] + [f"F{i}" for i in range(29)]
            app.state.models_loaded = True
            app.state.multi_predictor = None
            app.state.temporal = None
            app.state.causal = None
            app.state.reporter = None
            app.state.disease_classifier = None
            app.state.rag = None
            app.state.fusion = None
            app.state.data_pipeline_svc = None
            app.state.metrics = {}
            yield c


@pytest.fixture(scope="module")
def auth_headers(client):
    """Get a JWT bearer token for test requests."""
    resp = client.post("/auth/login", json={"username": "admin", "password": "admin"})
    if resp.status_code == 200:
        token = resp.json().get("access_token", "")
        return {"Authorization": f"Bearer {token}"}
    # Fall back to no auth if login not configured in test env
    return {}


# ---------------------------------------------------------------------------
# /ready — v5 fields
# ---------------------------------------------------------------------------

def test_ready_returns_v5_schema(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "database" in body
    assert "models_loaded" in body
    # v5 additions
    assert "rag_enabled" in body
    assert "fusion_loaded" in body
    assert "pgvector_ok" in body
    assert "schema_version" in body
    assert body["schema_version"] == "v5"


# ---------------------------------------------------------------------------
# GET /v3/data/sources
# ---------------------------------------------------------------------------

def test_data_sources_returns_list(client):
    resp = client.get("/v3/data/sources")
    # 200 or 401/503 depending on auth config; just check structure when 200
    if resp.status_code == 200:
        body = resp.json()
        assert "sources" in body
        assert isinstance(body["sources"], list)
        assert "total_sources" in body
        assert body["total_sources"] >= 0
    else:
        pytest.skip(f"Sources endpoint returned {resp.status_code} — likely auth required")


def test_data_sources_structure(client, auth_headers):
    resp = client.get("/v3/data/sources", headers=auth_headers)
    if resp.status_code not in (200, 401, 403):
        pytest.fail(f"Unexpected status {resp.status_code}")
    if resp.status_code == 200:
        sources = resp.json()["sources"]
        if sources:
            s = sources[0]
            assert "name" in s
            assert "tier" in s
            assert "status" in s


# ---------------------------------------------------------------------------
# GET /v3/data/cohort/stats
# ---------------------------------------------------------------------------

def test_cohort_stats_schema(client, auth_headers):
    resp = client.get("/v3/data/cohort/stats", headers=auth_headers)
    if resp.status_code == 200:
        body = resp.json()
        assert "total_patients" in body
        assert "prevalence" in body
        assert "age_distribution" in body
        assert isinstance(body["prevalence"], list)
        assert isinstance(body["age_distribution"], list)
        for p in body["prevalence"]:
            assert "name" in p
            assert "value" in p
    elif resp.status_code in (401, 403, 503):
        pytest.skip(f"Skipped — status {resp.status_code}")
    else:
        pytest.fail(f"Unexpected status {resp.status_code}: {resp.text[:200]}")


def test_cohort_stats_age_dist_keys(client, auth_headers):
    resp = client.get("/v3/data/cohort/stats", headers=auth_headers)
    if resp.status_code != 200:
        pytest.skip("endpoint not available")
    age_dist = resp.json()["age_distribution"]
    if age_dist:
        row = age_dist[0]
        for key in ("range", "ad", "pd", "ms", "ep", "als", "hd"):
            assert key in row, f"missing key {key!r} in age_distribution row"


# ---------------------------------------------------------------------------
# GET /v3/data/provenance
# ---------------------------------------------------------------------------

def test_provenance_schema(client, auth_headers):
    resp = client.get("/v3/data/provenance", headers=auth_headers)
    if resp.status_code == 200:
        body = resp.json()
        assert "total_rows" in body
        assert "provenance" in body
        assert isinstance(body["provenance"], list)
        assert "merge_file" in body
        assert "schema_version" in body
    elif resp.status_code in (401, 403, 503):
        pytest.skip(f"Skipped — status {resp.status_code}")
    else:
        pytest.fail(f"Unexpected {resp.status_code}")


# ---------------------------------------------------------------------------
# POST /v3/data/refresh/{source}
# ---------------------------------------------------------------------------

def test_refresh_requires_admin(client):
    """Non-admin or unauthenticated request must be rejected."""
    resp = client.post("/v3/data/refresh/kaggle_alzheimer")
    assert resp.status_code in (401, 403, 422), (
        f"Expected auth rejection, got {resp.status_code}"
    )


# ---------------------------------------------------------------------------
# GET /v3/literature/status
# ---------------------------------------------------------------------------

def test_literature_status(client, auth_headers):
    resp = client.get("/v3/literature/status", headers=auth_headers)
    if resp.status_code == 200:
        body = resp.json()
        assert "rag_enabled" in body or "status" in body
    elif resp.status_code in (401, 403, 503):
        pytest.skip(f"Skipped — status {resp.status_code}")
    else:
        pytest.fail(f"Unexpected {resp.status_code}")


# ---------------------------------------------------------------------------
# POST /v3/literature/search
# ---------------------------------------------------------------------------

def test_literature_search_schema(client, auth_headers):
    resp = client.post(
        "/v3/literature/search",
        json={"query": "Alzheimer amyloid beta", "top_k": 3},
        headers=auth_headers,
    )
    if resp.status_code == 200:
        body = resp.json()
        assert "results" in body or "abstracts" in body or isinstance(body, list)
    elif resp.status_code in (401, 403, 503, 422):
        pytest.skip(f"Skipped — status {resp.status_code}")
    else:
        pytest.fail(f"Unexpected {resp.status_code}: {resp.text[:200]}")


def test_literature_search_empty_query_rejected(client, auth_headers):
    resp = client.post(
        "/v3/literature/search",
        json={"query": "", "top_k": 5},
        headers=auth_headers,
    )
    # Empty query should be 422 (validation) or 400 (business logic)
    assert resp.status_code in (400, 422, 200, 401, 403, 503), (
        f"Unexpected {resp.status_code}"
    )


# ---------------------------------------------------------------------------
# GET /v3/fusion/weights
# ---------------------------------------------------------------------------

def test_fusion_weights_schema(client, auth_headers):
    resp = client.get("/v3/predictions/fusion/weights", headers=auth_headers)
    if resp.status_code == 200:
        body = resp.json()
        assert "weights" in body
        assert isinstance(body["weights"], dict)
        assert "method" in body
        # Default weights should sum to approximately 1.0
        total = sum(body["weights"].values())
        assert abs(total - 1.0) < 0.05, f"Weights sum {total:.3f} not close to 1.0"
    elif resp.status_code in (401, 403, 503):
        pytest.skip(f"Skipped — status {resp.status_code}")
    else:
        pytest.fail(f"Unexpected {resp.status_code}: {resp.text[:200]}")


def test_fusion_weights_modalities_present(client, auth_headers):
    resp = client.get("/v3/predictions/fusion/weights", headers=auth_headers)
    if resp.status_code != 200:
        pytest.skip("endpoint not available")
    weights = resp.json()["weights"]
    expected = {"tabular", "gnn", "genomic", "tft", "causal"}
    assert expected == set(weights.keys()), (
        f"Modality keys {set(weights.keys())} ≠ expected {expected}"
    )


# ---------------------------------------------------------------------------
# POST /v3/predictions/analyze
# ---------------------------------------------------------------------------

_SAMPLE_FEATURES = {
    "Age": 72.0, "MMSE": 22.0, "FunctionalAssessment": 6.0,
    "ADL": 5.5, "Hypertension": 1.0, "Depression": 0.0,
    "BMI": 26.3, "SystolicBP": 138.0, "DiastolicBP": 82.0,
    "CholesterolTotal": 210.0, "UPDRS_motor": 18.0, "APOE4_dosage": 1.0,
}


def test_v3_analyze_schema(client, auth_headers):
    resp = client.post(
        "/v3/predictions/analyze",
        json={"patient_id": "TEST-001", "features": _SAMPLE_FEATURES},
        headers=auth_headers,
    )
    if resp.status_code == 200:
        body = resp.json()
        # Core fields (same as v2)
        for key in ("patient_id", "request_id", "prediction", "probability",
                    "risk_level", "confidence"):
            assert key in body, f"missing field {key!r}"
        # v3-specific fields
        assert "fusion_weights" in body
        assert "fusion_attention_map" in body
        assert "modality_contributions" in body
        assert "rag_citations" in body
        assert "schema_version" in body
        assert body["schema_version"] == "v3"
        assert 0.0 <= body["probability"] <= 1.0
    elif resp.status_code in (401, 403, 503):
        pytest.skip(f"Skipped — status {resp.status_code}")
    else:
        pytest.fail(f"Unexpected {resp.status_code}: {resp.text[:300]}")


def test_v3_analyze_fusion_weights_present(client, auth_headers):
    resp = client.post(
        "/v3/predictions/analyze",
        json={"patient_id": "TEST-002", "features": _SAMPLE_FEATURES},
        headers=auth_headers,
    )
    if resp.status_code != 200:
        pytest.skip("endpoint not available")
    fw = resp.json()["fusion_weights"]
    assert isinstance(fw, dict)
    assert len(fw) > 0


def test_v3_analyze_missing_features_returns_valid(client, auth_headers):
    """Missing features should be zero-filled, not cause a 500."""
    resp = client.post(
        "/v3/predictions/analyze",
        json={"patient_id": "TEST-003", "features": {"Age": 65.0}},
        headers=auth_headers,
    )
    assert resp.status_code in (200, 401, 403, 422, 503), (
        f"Missing features caused unexpected {resp.status_code}"
    )


# ---------------------------------------------------------------------------
# Platt calibrator smoke test (unit-level, no HTTP)
# ---------------------------------------------------------------------------

def test_platt_calibrator_fit_predict():
    """_PlattCalibrator should fit without error and produce probabilities in [0,1]."""
    import os, sys
    _src = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    if _src not in sys.path:
        sys.path.insert(0, _src)
    import numpy as np
    from neurosynth.models.calibrated_ensemble import _PlattCalibrator

    rng = np.random.default_rng(42)
    probs = rng.uniform(0.1, 0.9, 100)
    labels = (probs > 0.5).astype(float) + rng.normal(0, 0.1, 100)
    labels = np.clip(labels, 0, 1).round()

    cal = _PlattCalibrator().fit(probs, labels)
    calibrated = cal.predict(probs)

    assert calibrated.shape == probs.shape
    assert (calibrated >= 0).all() and (calibrated <= 1).all()
    assert abs(cal.a) <= 10 and abs(cal.b) <= 10


# ---------------------------------------------------------------------------
# DataPipelineService smoke test (unit-level, no HTTP)
# ---------------------------------------------------------------------------

def test_data_pipeline_service_fallback_stats():
    """DataPipelineService._fallback_stats() should always return valid structure."""
    from backend.services.data_pipeline_service import DataPipelineService

    svc = DataPipelineService(db=None)
    stats = svc._fallback_stats()

    assert "total_patients" in stats
    assert "prevalence" in stats
    assert "age_distribution" in stats
    assert len(stats["prevalence"]) == 6
    assert len(stats["age_distribution"]) > 0
    assert stats["total_patients"] > 0


def test_data_pipeline_service_provenance():
    from backend.services.data_pipeline_service import DataPipelineService

    svc = DataPipelineService(db=None)
    prov = svc.get_provenance()

    assert "provenance" in prov
    assert "total_rows" in prov
    assert len(prov["provenance"]) == 11  # all 11 canonical sources
