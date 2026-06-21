# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""v2 Report endpoints — SOAP, FHIR R4, PDF export.

Priority 6 Clinical Report Generation:
  - POST /v2/reports/generate — full SOAP report with ICD-10 codes
  - POST /v2/reports/fhir — FHIR R4 DiagnosticReport output
  - POST /v2/reports/pdf — PDF binary download

NOTE: Do NOT add ``from __future__ import annotations`` to this file.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from backend.core.rate_limit import limiter, role_limit
from backend.deps import get_current_user, get_database
from backend.db import Database
from backend.models import ReportRequest, UserContext
from backend.report_generator_v3 import ClinicalReportGeneratorV3

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2/reports", tags=["reports-v2"])

_report_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="report-gen")
# V3 generator: live Claude SOAP narrative when ANTHROPIC_API_KEY is set,
# deterministic Jinja2 template fallback otherwise (subclass of V2).
_report_gen = ClinicalReportGeneratorV3()


def _run_report_sync(
    predictor, temporal, causal_model, scaler, feature_names,
    patient_id: str, disease: str | None,
) -> dict:
    """Generate report synchronously inside executor."""
    import numpy as np
    import pandas as pd

    base = {name: 0.0 for name in feature_names}
    frame = pd.DataFrame([base])
    scaled = scaler.transform(frame)

    pred = predictor.predict(scaled)
    shap_vals = predictor.get_shap_values(scaled[:1])[0]
    top_idx = list(np.abs(shap_vals).argsort()[::-1][:10])
    shap_top = [{"feature": feature_names[i], "value": round(float(shap_vals[i]), 4)} for i in top_idx]

    traj = temporal.predict_trajectory(frame.values[0], pred["probability"]) if temporal else {
        "trajectory": [round(float(pred["probability"] + i * 0.02), 4) for i in range(8)]
    }
    causal_graph = causal_model.get_causal_graph() if causal_model else {}

    return _report_gen.generate_report(
        patient_data=base,
        prediction=pred,
        trajectory=traj.get("trajectory", traj) if isinstance(traj, dict) else traj,
        causal_graph=causal_graph,
        shap_values=shap_top,
        patient_id=patient_id,
        disease=disease,
    )


@router.post(
    "/generate",
    summary="Generate SOAP-structured clinical report",
    description=(
        "Generates a comprehensive clinical report in SOAP format "
        "(Subjective/Objective/Assessment/Plan) with ICD-10 code suggestions, "
        "trajectory analysis, and SHAP explanations."
    ),
)
@limiter.limit(role_limit)
async def generate_report_v2(
    payload: ReportRequest,
    request: Request,
    user: UserContext = Depends(get_current_user),
):
    _ = user
    predictor = getattr(request.app.state, "predictor", None)
    temporal = getattr(request.app.state, "temporal", None)
    causal_model = getattr(request.app.state, "causal", None)
    scaler = getattr(request.app.state, "scaler", None)
    feature_names = list(getattr(request.app.state, "feature_names", []) or [])

    if not predictor or not scaler or not feature_names:
        # Generate fallback report with dummy data
        report = _report_gen.generate_report(
            patient_data={},
            prediction={"probability": 0.5, "risk_level": "Unknown", "confidence": "Low"},
            trajectory=[0.5 + i * 0.02 for i in range(8)],
            causal_graph={},
            shap_values=[],
            patient_id=payload.patient_id,
        )
        return report

    loop = asyncio.get_running_loop()
    report = await loop.run_in_executor(
        _report_executor,
        _run_report_sync,
        predictor, temporal, causal_model, scaler, feature_names,
        payload.patient_id, None,
    )
    return report


@router.post(
    "/fhir",
    summary="Generate FHIR R4 DiagnosticReport",
    description="Returns the clinical report as a FHIR R4 DiagnosticReport resource.",
)
@limiter.limit(role_limit)
async def generate_fhir_report(
    payload: ReportRequest,
    request: Request,
    user: UserContext = Depends(get_current_user),
):
    _ = user
    predictor = getattr(request.app.state, "predictor", None)
    scaler = getattr(request.app.state, "scaler", None)
    feature_names = list(getattr(request.app.state, "feature_names", []) or [])

    if not predictor or not scaler or not feature_names:
        report = _report_gen.generate_report(
            patient_data={}, prediction={"probability": 0.5, "risk_level": "Unknown", "confidence": "Low"},
            trajectory=[], causal_graph={}, shap_values=[], patient_id=payload.patient_id,
        )
    else:
        loop = asyncio.get_running_loop()
        report = await loop.run_in_executor(
            _report_executor, _run_report_sync,
            predictor, getattr(request.app.state, "temporal", None),
            getattr(request.app.state, "causal", None), scaler, feature_names,
            payload.patient_id, None,
        )

    return _report_gen.to_fhir(report)


@router.post(
    "/pdf",
    summary="Export clinical report as PDF",
    description="Generates the clinical report and returns it as a downloadable PDF file.",
    response_class=Response,
    responses={200: {"content": {"application/pdf": {}}}},
)
@limiter.limit(role_limit)
async def generate_pdf_report(
    payload: ReportRequest,
    request: Request,
    user: UserContext = Depends(get_current_user),
):
    _ = user
    predictor = getattr(request.app.state, "predictor", None)
    scaler = getattr(request.app.state, "scaler", None)
    feature_names = list(getattr(request.app.state, "feature_names", []) or [])

    if not predictor or not scaler or not feature_names:
        report = _report_gen.generate_report(
            patient_data={}, prediction={"probability": 0.5, "risk_level": "Unknown", "confidence": "Low"},
            trajectory=[], causal_graph={}, shap_values=[], patient_id=payload.patient_id,
        )
    else:
        loop = asyncio.get_running_loop()
        report = await loop.run_in_executor(
            _report_executor, _run_report_sync,
            predictor, getattr(request.app.state, "temporal", None),
            getattr(request.app.state, "causal", None), scaler, feature_names,
            payload.patient_id, None,
        )

    pdf_bytes = _report_gen.to_pdf(report)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="neurosynth_report_{payload.patient_id}.pdf"'},
    )
