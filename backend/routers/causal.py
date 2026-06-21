# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from backend.core.rate_limit import limiter, role_limit
from backend.deps import get_current_user
from backend.models import CausalRequest, CausalResponse, UserContext
from backend.tasks import causal_analysis

router = APIRouter(prefix="/causal", tags=["causal"])


class SimulateRequest(BaseModel):
    patient_data: dict
    variable: str
    new_value: float


@router.post(
    "/analyze",
    response_model=CausalResponse,
    summary="Queue causal analysis",
    description="Queues causal analysis for patient features and intervention simulation.",
)
@limiter.limit(role_limit)
async def analyze_causal(payload: CausalRequest, request: Request, user: UserContext = Depends(get_current_user)) -> CausalResponse:
    _ = request
    _ = user
    task = causal_analysis.delay(payload.patient_id)
    return CausalResponse(task_id=task.id, patient_id=payload.patient_id, status="queued")


@router.post("/simulate")
async def simulate_intervention(
    payload: SimulateRequest,
    request: Request,
    user: UserContext = Depends(get_current_user),
):
    _ = user
    causal_model = getattr(request.app.state, "causal", None)
    if causal_model is None:
        raise HTTPException(status_code=503, detail="Causal model not loaded")
    return causal_model.simulate_intervention(
        variable=payload.variable,
        new_value_normalized=payload.new_value,
        patient_data=payload.patient_data,
    )


@router.get("/graph")
async def get_causal_graph(request: Request, user: UserContext = Depends(get_current_user)):
    _ = user
    causal_model = getattr(request.app.state, "causal", None)
    if causal_model is None:
        return {}
    return causal_model.get_causal_graph()
