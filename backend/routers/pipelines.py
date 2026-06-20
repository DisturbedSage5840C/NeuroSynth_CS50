from fastapi import APIRouter, Depends, Request

from backend.core.rate_limit import limiter, role_limit
from backend.core.security import Role
from backend.deps import require_role
from backend.models import PipelineRunRequest, PipelineRunResponse, UserContext
from backend.services.kubeflow_service import trigger_training_run

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
router = APIRouter(prefix="/pipelines", tags=["pipelines"])


@router.post(
    "/run",
    response_model=PipelineRunResponse,
    summary="Trigger Kubeflow training pipeline",
    description="Starts a Kubeflow pipeline run and returns a run ID for polling.",
)
@limiter.limit(role_limit)
async def run_pipeline(
    payload: PipelineRunRequest,
    request: Request,
    user: UserContext = Depends(require_role(Role.ADMIN, Role.RESEARCHER)),
) -> PipelineRunResponse:
    _ = request
    _ = user
    run_id = trigger_training_run(project_name=payload.project_name, model_name=payload.model_name)
    return PipelineRunResponse(run_id=run_id)
