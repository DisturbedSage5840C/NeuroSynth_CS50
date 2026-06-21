# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
import asyncio
import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis

from backend.core.metrics import ACTIVE_SSE_CONNECTIONS
from backend.core.rate_limit import limiter, role_limit
from backend.deps import get_current_user, get_redis
from backend.models import SSEHandshakeResponse, UserContext
from backend.wearable_simulator import WearableSimulator

router = APIRouter(prefix="/biomarkers", tags=["biomarkers"])


@router.get(
    "",
    response_model=SSEHandshakeResponse,
    summary="Biomarker SSE endpoint discovery",
    description="Provides the biomarker stream path for frontend auto-discovery.",
)
@limiter.limit(role_limit)
async def biomarkers_info(request: Request, user: UserContext = Depends(get_current_user)) -> SSEHandshakeResponse:
    _ = request
    _ = user
    return SSEHandshakeResponse(stream="/biomarkers/stream")


@router.get(
    "/stream",
    response_class=StreamingResponse,
    summary="Biomarker live stream",
    description="Server-Sent Events stream for real-time phase progress emitted from Redis pub/sub.",
)
@limiter.limit(role_limit)
async def biomarkers_stream(
    request: Request,
    user: UserContext = Depends(get_current_user),
    redis_client: Redis = Depends(get_redis),
):
    _ = request
    _ = user

    async def event_generator() -> asyncio.AsyncGenerator[str, None]:
        pubsub = redis_client.pubsub()
        await pubsub.subscribe("biomarkers.progress")
        ACTIVE_SSE_CONNECTIONS.inc()
        try:
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=5.0)
                if message and message.get("data"):
                    payload = message["data"]
                    if isinstance(payload, (bytes, bytearray)):
                        payload = payload.decode("utf-8")
                    json.loads(payload)
                    yield f"event: progress\ndata: {payload}\n\n"
                else:
                    yield "event: heartbeat\ndata: {}\n\n"
                await asyncio.sleep(0.25)
        finally:
            ACTIVE_SSE_CONNECTIONS.dec()
            await pubsub.unsubscribe("biomarkers.progress")
            await pubsub.close()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get(
    "/live/{patient_id}",
    response_class=StreamingResponse,
    summary="Wearable live stream",
    description="Server-Sent Events stream for simulated wearable vitals generated with AR(1) autocorrelation.",
)
@limiter.limit(role_limit)
async def live_biomarkers(
    patient_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
):
    _ = request
    _ = user
    simulator = WearableSimulator(seed=abs(hash(patient_id)) % (2**32))

    async def event_generator() -> asyncio.AsyncGenerator[str, None]:
        ACTIVE_SSE_CONNECTIONS.inc()
        try:
            while True:
                payload = simulator.next_reading(patient_id=patient_id)
                yield f"event: vitals\ndata: {json.dumps(payload)}\n\n"
                await asyncio.sleep(2.0)
        finally:
            ACTIVE_SSE_CONNECTIONS.dec()

    return StreamingResponse(event_generator(), media_type="text/event-stream")
