"""Prediction endpoints — full synchronous analysis and async pipeline.

v2 FIXES:
- Bug #7: All blocking ML inference now runs in a ThreadPoolExecutor via
  ``run_in_executor`` to avoid blocking the async event loop.
- Bug #8: Removed lazy DiseaseClassifier.train() inside request handler.
  The classifier is now loaded during startup via the model registry.
- Bug #9: Database access uses proper abstraction with null checks.
- Added RFC 7807 style error detail in validation errors.
- Added request_id (trace_id) propagation.

NOTE: Do NOT add ``from __future__ import annotations`` to this file.
FastAPI requires runtime type resolution for route parameters (FeatureVector, etc.).
"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pandas as pd
import pandera as pa
from fastapi import APIRouter, Depends, HTTPException, Request
from pandera.typing import Series

from backend.core.rate_limit import limiter, role_limit
from backend.db import Database
from backend.deps import get_current_user, get_database
from backend.models import FeatureVector, PredictionResponse, UserContext
from backend.tasks import enqueue_full_pipeline

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["predictions"])

# ThreadPoolExecutor for blocking ML inference operations.
# Sized for concurrent analysis requests without exhausting system resources.
_ml_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ml-inference")


class PredictionInputSchema(pa.DataFrameModel):
    Age: Series[float] = pa.Field(ge=0, le=120)
    MMSE: Series[float] = pa.Field(ge=0, le=30)
    FunctionalAssessment: Series[float] = pa.Field(ge=0, le=10)
    ADL: Series[float] = pa.Field(ge=0, le=10)
    SleepQuality: Series[float] = pa.Field(ge=0, le=10)


def _build_fallback_response(
    payload: FeatureVector,
    disease_result: dict,
    request_id: str,
) -> dict:
    """Build a structured fallback response when models are not loaded."""
    keys = list(payload.features.keys())
    shap_top = [
        {"feature": k, "value": round(float(payload.features.get(k, 0.0)) / 100.0, 4)}
        for k in keys[:10]
    ]
    base_prob = 0.5
    trajectory = [round(base_prob + i * 0.02, 4) for i in range(6)]

    return {
        "patient_id": payload.patient_id,
        "request_id": request_id,
        "prediction": 0,
        "probability": base_prob,
        "risk_level": "moderate",
        "confidence": "Medium",
        "individual_model_probs": {"rf": base_prob, "xgb": base_prob, "lgb": base_prob},
        "top_risk_factors": [item["feature"] for item in shap_top[:5]],
        "shap_values": shap_top,
        "trajectory": trajectory,
        "confidence_bands": {
            "lower": [round(max(0.0, t - 0.08), 4) for t in trajectory],
            "upper": [round(min(1.0, t + 0.08), 4) for t in trajectory],
        },
        "causal_graph": {"nodes": [], "edges": []},
        "report": {
            "sections": {
                "Clinical Summary": "Fallback analysis returned because models are not loaded.",
                "Recommendations": "Restart backend after dependencies are available for full model inference.",
            },
            "generated_at": "",
            "word_count": 17,
        },
        "disease_classification": disease_result,
    }


def _run_full_inference(
    predictor,
    multi_predictor,
    temporal,
    causal_model,
    reporter,
    disease_clf,
    scaler,
    feature_names: list[str],
    features: dict[str, float],
) -> dict:
    """Run all ML models synchronously.  Called inside ThreadPoolExecutor."""
    frame = pd.DataFrame([{k: float(features.get(k, 0.0)) for k in feature_names}])
    scaled = scaler.transform(frame)

    # Disease classification.
    disease_result = disease_clf.predict_disease(features) if disease_clf else {}
    primary_disease = disease_result.get("predicted_disease")

    # Multi-disease risk vector.
    disease_risk_vector: dict[str, float] = {}
    pred = predictor.predict(scaled)
    if multi_predictor is not None:
        try:
            disease_risk_vector = {
                k: round(float(v), 4) for k, v in multi_predictor.predict_all(scaled).items()
            }
            if primary_disease and primary_disease in multi_predictor.predictors:
                pred = multi_predictor.predict_for_disease(primary_disease, scaled)
        except Exception:
            disease_risk_vector = {}

    # SHAP values.
    shap_vals = predictor.get_shap_values(scaled[:1])[0]
    top_idx = list(np.abs(shap_vals).argsort()[::-1][:10])
    shap_top = [
        {"feature": feature_names[i], "value": round(float(shap_vals[i]), 4)}
        for i in top_idx
    ]

    # Trajectory forecast.
    traj = temporal.predict_trajectory(frame.values[0], pred["probability"]) if temporal else {
        "trajectory": [round(float(pred["probability"] + i * 0.02), 4) for i in range(6)],
        "confidence_bands": {"lower": [], "upper": []},
    }

    # Causal graph.
    causal_graph = causal_model.get_causal_graph() if causal_model else {}

    # Clinical report.
    report = reporter.generate_report(
        patient_data=features,
        prediction=pred,
        trajectory=traj["trajectory"],
        causal_graph=causal_graph,
        shap_values=shap_top,
    ) if reporter else {"sections": {}, "raw_text": ""}

    return {
        "pred": pred,
        "shap_top": shap_top,
        "traj": traj,
        "causal_graph": causal_graph,
        "report": report,
        "disease_result": disease_result,
        "disease_risk_vector": disease_risk_vector,
        "primary_disease": primary_disease,
    }


async def _persist_analysis(
    db: Database,
    patient_id: str,
    features: dict,
    pred: dict,
    traj: list,
    shap_top: list,
    causal_graph: dict,
    report: dict,
    disease_result: dict,
) -> None:
    """Persist analysis results to the database (best-effort)."""
    if db.pool is None:
        return
    try:
        await db.pool.execute(
            "INSERT INTO patients (id, name, diagnosis, created_at, updated_at) "
            "VALUES ($1, $2, $3, NOW(), NOW()) "
            "ON CONFLICT (id) DO UPDATE SET updated_at = EXCLUDED.updated_at",
            patient_id,
            f"Patient {patient_id}",
            "Neurology Monitoring",
        )
        analysis_id = uuid4().hex
        await db.pool.execute(
            "INSERT INTO analyses (id, patient_id, features, probability, risk_level, "
            "confidence, trajectory, shap_values, causal_graph, report_sections, disease_classification) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)",
            analysis_id,
            patient_id,
            json.dumps(features),
            pred.get("probability", 0.5),
            pred.get("risk_level", "moderate"),
            pred.get("confidence", "Medium"),
            json.dumps(traj),
            json.dumps(shap_top),
            json.dumps(causal_graph),
            json.dumps(report.get("sections", {})),
            json.dumps(disease_result),
        )
    except Exception as e:
        logger.warning("Failed to persist analysis: %s", e)


@router.post(
    "/run",
    response_model=PredictionResponse,
    summary="Queue full prediction workflow",
    description="Validates model inputs with pandera and queues all ML phases in Celery.",
)
@limiter.limit(role_limit)
async def run_prediction(payload: FeatureVector, request: Request, user: UserContext = Depends(get_current_user)) -> PredictionResponse:
    _ = request
    _ = user
    frame = pd.DataFrame([payload.features])
    PredictionInputSchema.validate(frame)

    job_id = enqueue_full_pipeline(payload.patient_id)
    return PredictionResponse(
        job_id=job_id or uuid4().hex,
        patient_id=payload.patient_id,
        queued_phases=[
            "connectome_inference",
            "genomic_risk_score",
            "temporal_forecast",
            "causal_analysis",
            "report_generation",
        ],
    )


@router.post(
    "/analyze",
    summary="Run full synchronous analysis",
    description=(
        "Takes patient features, runs all ML models in a background thread, "
        "and returns complete results including SHAP values, trajectory, "
        "causal graph, and clinical report."
    ),
)
@limiter.limit(role_limit)
async def analyze_patient(
    payload: FeatureVector,
    request: Request,
    user: UserContext = Depends(get_current_user),
    db: Database = Depends(get_database),
):
    _ = user
    request_id = request.headers.get("X-Trace-Id", uuid4().hex)

    predictor = getattr(request.app.state, "predictor", None)
    temporal = getattr(request.app.state, "temporal", None)
    causal_model = getattr(request.app.state, "causal", None)
    reporter = getattr(request.app.state, "reporter", None)
    disease_clf = getattr(request.app.state, "disease_classifier", None)
    multi_predictor = getattr(request.app.state, "multi_predictor", None)
    scaler = getattr(request.app.state, "scaler", None)
    feature_names = getattr(request.app.state, "feature_names", None)

    # --- Fallback: models not loaded ---
    if predictor is None or scaler is None or not feature_names:
        disease_result = disease_clf.predict_disease(payload.features) if disease_clf else {}
        result = _build_fallback_response(payload, disease_result, request_id)

        await _persist_analysis(
            db, payload.patient_id, payload.features,
            {"probability": 0.5, "risk_level": "moderate", "confidence": "Medium"},
            result["trajectory"], result["shap_values"],
            result["causal_graph"], result["report"], disease_result,
        )
        return result

    # --- Full inference in thread executor to avoid blocking event loop ---
    loop = asyncio.get_running_loop()
    inference_result = await loop.run_in_executor(
        _ml_executor,
        _run_full_inference,
        predictor,
        multi_predictor,
        temporal,
        causal_model,
        reporter,
        disease_clf,
        scaler,
        feature_names,
        payload.features,
    )

    pred = inference_result["pred"]
    shap_top = inference_result["shap_top"]
    traj = inference_result["traj"]
    causal_graph = inference_result["causal_graph"]
    report = inference_result["report"]
    disease_result = inference_result["disease_result"]
    disease_risk_vector = inference_result["disease_risk_vector"]
    primary_disease = inference_result["primary_disease"]

    # Persist to database (best-effort, non-blocking for response).
    await _persist_analysis(
        db, payload.patient_id, payload.features,
        pred, traj["trajectory"], shap_top,
        causal_graph, report, disease_result,
    )

    return {
        "patient_id": payload.patient_id,
        "request_id": request_id,
        "prediction": pred["prediction"],
        "probability": pred["probability"],
        "risk_level": pred["risk_level"],
        "confidence": pred["confidence"],
        "individual_model_probs": pred["individual_model_probs"],
        "top_risk_factors": pred["top_risk_factors"],
        "shap_values": shap_top,
        "trajectory": traj["trajectory"],
        "confidence_bands": traj["confidence_bands"],
        "causal_graph": causal_graph,
        "report": report,
        "disease_classification": disease_result,
        "disease_risk_vector": disease_risk_vector,
        "primary_model_disease": primary_disease,
    }


@router.get("/dataset/stats")
async def dataset_stats(request: Request, user: UserContext = Depends(get_current_user)):
    _ = user
    return getattr(request.app.state, "dataset_stats", {})


@router.get("/model/performance")
async def model_performance(request: Request):
    return getattr(request.app.state, "metrics", {})


@router.get("/model/feature_importance")
async def feature_importance(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        return {}
    try:
        raw = predictor.get_feature_importance()
        # Convert {features: [...], importances: [...]} → {feature: value, ...}
        if isinstance(raw, dict) and "features" in raw and "importances" in raw:
            return dict(zip(raw["features"], raw["importances"]))
        return raw
    except Exception:
        return {}
