from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response
from redis.asyncio import Redis

from backend.celery_app import celery_app
from backend.core.metrics import CELERY_QUEUE_DEPTH, render_metrics
from backend.core.rate_limit import limiter, role_limit
from backend.core.security import Role
from backend.deps import get_current_user, get_redis, require_role
from backend.models import CeleryTaskResult, QueueDepthResponse, UserContext

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
router = APIRouter(prefix="/admin", tags=["admin"])


@router.get(
    "/queue-depth",
    response_model=QueueDepthResponse,
    summary="Celery queue depth",
    description="Returns current queue depth from Redis list length and updates Prometheus gauge.",
)
@limiter.limit(role_limit)
async def queue_depth(
    request: Request,
    user: UserContext = Depends(require_role(Role.ADMIN)),
    redis_client: Redis = Depends(get_redis),
) -> QueueDepthResponse:
    _ = request
    _ = user
    depth = int(await redis_client.llen("celery"))
    CELERY_QUEUE_DEPTH.labels(queue="celery").set(depth)
    return QueueDepthResponse(queue="celery", depth=depth)


@router.get(
    "/tasks/{task_id}",
    response_model=CeleryTaskResult,
    summary="Get Celery task status",
    description="Returns current Celery task state and result metadata for debugging and observability.",
)
@limiter.limit(role_limit)
async def task_status(task_id: str, request: Request, user: UserContext = Depends(require_role(Role.ADMIN))) -> CeleryTaskResult:
    _ = request
    _ = user
    async_result = celery_app.AsyncResult(task_id)
    phase = str((async_result.result or {}).get("phase", "unknown")) if isinstance(async_result.result, dict) else "unknown"
    meta = async_result.result if isinstance(async_result.result, dict) else {}
    return CeleryTaskResult(task_id=task_id, phase=phase, state=async_result.state, meta=meta)


@router.get(
    "/metrics",
    summary="Prometheus metrics (admin)",
    description="Admin-scoped endpoint that exposes Prometheus-formatted metrics payload.",
)
@limiter.limit(role_limit)
async def metrics_endpoint(request: Request, user: UserContext = Depends(require_role(Role.ADMIN))) -> Response:
    _ = request
    _ = user
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)
