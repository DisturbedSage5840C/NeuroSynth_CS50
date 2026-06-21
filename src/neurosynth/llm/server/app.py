# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, timedelta, timezone

import asyncpg
import jwt
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from neurosynth.llm.generation import ConstrainedReportGenerator


SECRET = os.getenv("NEURO_LLM_JWT_SECRET", "dev-secret")
DB_DSN = os.getenv("NEURO_TIMESCALE_DSN", "postgresql://postgres:postgres@timescaledb:5432/neurosynth")

app = FastAPI(title="NeuroSynth LLM Service")
limiter = Limiter(key_func=get_remote_address, default_limits=["10/second"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: PlainTextResponse("rate limit exceeded", status_code=429))

REQUEST_COUNT = Counter("neuro_llm_requests_total", "Total report requests")
REQUEST_LAT = Histogram("neuro_llm_request_latency_seconds", "Request latency")

_generator = ConstrainedReportGenerator()
_pool: asyncpg.Pool | None = None


class PatientContext(BaseModel):
    patient_context: dict


def verify_bearer(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, SECRET, algorithms=["HS256"])
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))
    exp = payload.get("exp")
    if exp and datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Token expired")
    return payload


@app.on_event("startup")
async def startup() -> None:
    global _pool
    _pool = await asyncpg.create_pool(DB_DSN, min_size=1, max_size=4)
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_inference_logs (
                ts TIMESTAMPTZ NOT NULL,
                prompt_hash TEXT,
                latency_seconds DOUBLE PRECISION,
                output_tokens INT,
                plausibility_score DOUBLE PRECISION
            );
            """
        )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.post("/generate_report")
@limiter.limit("10/second")
async def generate_report(payload: PatientContext, auth=Depends(verify_bearer)) -> dict:
    _ = auth
    REQUEST_COUNT.inc()
    t0 = time.time()

    prompt = str(payload.patient_context)
    out = _generator.generate_report(prompt)

    out_tokens = len(str(out).split())
    REQUEST_LAT.observe(time.time() - t0)

    if _pool is not None:
        async with _pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO llm_inference_logs (ts, prompt_hash, latency_seconds, output_tokens, plausibility_score)
                VALUES ($1, $2, $3, $4, $5)
                """,
                datetime.now(timezone.utc),
                out.get("prompt_hash") or hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                float(out.get("latency_seconds", 0.0)),
                int(out_tokens),
                float(out.get("plausibility_score", 0.0)),
            )

    return out


def issue_token(sub: str) -> str:
    payload = {"sub": sub, "exp": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp())}
    return jwt.encode(payload, SECRET, algorithm="HS256")
