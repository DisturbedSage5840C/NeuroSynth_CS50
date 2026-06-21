# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from fastapi import APIRouter, Depends, Request

from backend.core.rate_limit import limiter, role_limit
from backend.deps import get_current_user
from backend.models import ReportRequest, ReportResponse, UserContext
from backend.tasks import report_generation

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post(
    "/generate",
    response_model=ReportResponse,
    summary="Queue report generation",
    description="Queues the report generation phase and returns Celery task ID for polling.",
)
@limiter.limit(role_limit)
async def generate_report(payload: ReportRequest, request: Request, user: UserContext = Depends(get_current_user)) -> ReportResponse:
    _ = request
    _ = user
    predictor = getattr(request.app.state, "predictor", None)
    temporal = getattr(request.app.state, "temporal", None)
    causal_model = getattr(request.app.state, "causal", None)
    reporter = getattr(request.app.state, "reporter", None)
    scaler = getattr(request.app.state, "scaler", None)
    feature_names = list(getattr(request.app.state, "feature_names", []) or [])

    if predictor is not None and temporal is not None and reporter is not None and scaler is not None and feature_names:
        import pandas as pd

        base = {name: 0.0 for name in feature_names}
        frame = pd.DataFrame([base])
        scaled = scaler.transform(frame)
        pred = predictor.predict(scaled)
        shap_vals = predictor.get_shap_values(scaled[:1])[0]
        top_idx = list(abs(shap_vals).argsort()[::-1][:10])
        shap_top = [{"feature": feature_names[i], "value": round(float(shap_vals[i]), 4)} for i in top_idx]
        traj = temporal.predict_trajectory(frame.values[0], pred["probability"])
        causal_graph = causal_model.get_causal_graph() if causal_model is not None else {}
        report = reporter.generate_report(
            patient_data=base,
            prediction=pred,
            trajectory=traj["trajectory"],
            causal_graph=causal_graph,
            shap_values=shap_top,
        )
        return ReportResponse(task_id="sync", patient_id=payload.patient_id, status="completed", report=report)

    task = report_generation.delay(payload.patient_id, payload.notes)
    return ReportResponse(task_id=task.id, patient_id=payload.patient_id, status="queued")
