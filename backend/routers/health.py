from fastapi import APIRouter, Depends, Request

from backend.db import Database
from backend.deps import get_database
from backend.models import HealthResponse, ReadyResponse

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
router = APIRouter(prefix="", tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns process liveness for container orchestration health checks.",
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness probe",
    description="Checks PostgreSQL, Redis, ML models, RAG, CrossAttentionFusion, and pgvector.",
)
async def ready(request: Request, db: Database = Depends(get_database)) -> ReadyResponse:
    db_ok = False
    redis_ok = False
    pgvector_ok = False
    models_loaded = bool(getattr(request.app.state, "predictor", None))
    rag_enabled = bool(getattr(request.app.state, "rag", None) and
                       getattr(request.app.state.rag, "enabled", False))
    fusion_loaded = bool(getattr(request.app.state, "fusion", None))
    redis_client = getattr(request.app.state, "redis", None)

    try:
        row = await db.fetchrow("SELECT 1 AS ok")
        db_ok = row is not None and row["ok"] == 1
    except Exception:
        db_ok = False

    try:
        redis_ok = bool(redis_client and await redis_client.ping())
    except Exception:
        redis_ok = False

    # Verify pgvector is usable — table present + at least one embedding row queryable
    if db_ok:
        try:
            count_row = await db.fetchrow(
                "SELECT COUNT(*) AS n FROM literature_embeddings LIMIT 1"
            )
            pgvector_ok = count_row is not None
        except Exception:
            pgvector_ok = False

    is_ready = db_ok and models_loaded
    startup_error = getattr(request.app.state, "startup_error", "")
    return ReadyResponse(
        status="ready" if is_ready else "degraded",
        database=db_ok,
        redis=redis_ok,
        models_loaded=models_loaded,
        rag_enabled=rag_enabled,
        fusion_loaded=fusion_loaded,
        pgvector_ok=pgvector_ok,
        schema_version="v5",
        startup_error=startup_error,
    )
