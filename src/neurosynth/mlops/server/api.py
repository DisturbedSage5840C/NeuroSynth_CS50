from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

import asyncpg
import jwt
import structlog
from celery import Celery
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jose import jwk
from jose.utils import base64url_decode
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel


LOGGER = structlog.get_logger(__name__)
JWKS_URL = os.getenv("NEURO_JWKS_URL", "")
API_ALLOWED_ORIGINS = [x for x in os.getenv("NEURO_ALLOWED_ORIGINS", "http://localhost").split(",") if x]
TIMESCALE_DSN = os.getenv("NEURO_TIMESCALE_DSN", "postgresql://postgres:postgres@timescaledb:5432/neurosynth")

celery_app = Celery("neurosynth", broker=os.getenv("NEURO_REDIS_URL", "redis://localhost:6379/0"), backend=os.getenv("NEURO_REDIS_URL", "redis://localhost:6379/0"))


class PatientAnalysisRequest(BaseModel):
    patient_id: str
    analysis_config: dict[str, Any]


class AnalysisJobResponse(BaseModel):
    job_id: str
    status: str


REQUIRED_MODALITIES = ["imaging", "biomarker"]


app = FastAPI(title="NeuroSynth API", version="v1")
app.add_middleware(CORSMiddleware, allow_origins=API_ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

_rate = defaultdict(lambda: deque())
_pool: asyncpg.Pool | None = None


@app.on_event("startup")
async def startup() -> None:
    global _pool
    _pool = await asyncpg.create_pool(TIMESCALE_DSN, min_size=1, max_size=5)
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS api_request_log (
                ts TIMESTAMPTZ,
                endpoint TEXT,
                method TEXT,
                latency_ms DOUBLE PRECISION,
                status INT,
                api_key_hash TEXT
            );
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_registry (
                patient_id TEXT PRIMARY KEY,
                active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_data_inventory (
                patient_id TEXT NOT NULL,
                modality TEXT NOT NULL,
                available BOOLEAN NOT NULL DEFAULT TRUE,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY(patient_id, modality)
            );
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_jobs (
                job_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                submitted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                status TEXT NOT NULL,
                config JSONB,
                result JSONB
            );
            """
        )


async def _validate_patient_and_data(patient_id: str, analysis_config: dict[str, Any]) -> None:
    if _pool is None:
        raise HTTPException(status_code=503, detail="Database pool unavailable")

    required_modalities = analysis_config.get("required_modalities", REQUIRED_MODALITIES)
    async with _pool.acquire() as conn:
        patient_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM patient_registry WHERE patient_id=$1 AND active=TRUE)",
            patient_id,
        )
        if not patient_exists:
            raise HTTPException(status_code=404, detail="Patient not found")

        for modality in required_modalities:
            available = await conn.fetchval(
                "SELECT available FROM patient_data_inventory WHERE patient_id=$1 AND modality=$2",
                patient_id,
                str(modality),
            )
            if not available:
                raise HTTPException(status_code=409, detail=f"Missing required modality: {modality}")


async def _fetch_jwks() -> dict:
    import httpx

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
    if not JWKS_URL:
        raise HTTPException(status_code=503, detail="JWKS endpoint is not configured")

    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.get(JWKS_URL)
        r.raise_for_status()
        return r.json()


async def verify_api_key(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth.split(" ", 1)[1]

    jwks = await _fetch_jwks()
    headers = jwt.get_unverified_header(token)
    kid = headers.get("kid")
    key = next((k for k in jwks.get("keys", []) if k.get("kid") == kid), None)
    if key is None:
        raise HTTPException(status_code=401, detail="Unknown key id")

    message, encoded_sig = token.rsplit(".", 1)
    decoded_sig = base64url_decode(encoded_sig.encode("utf-8"))
    public_key = jwk.construct(key)
    if not public_key.verify(message.encode("utf-8"), decoded_sig):
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = jwt.decode(token, options={"verify_signature": False})
    if payload.get("exp", 0) < int(time.time()):
        raise HTTPException(status_code=401, detail="Token expired")
    return payload


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start = time.time()
    resp = await call_next(request)
    dur = (time.time() - start) * 1000.0

    api_hash = str(hash(request.headers.get("Authorization", "")))
    LOGGER.info("http.request", endpoint=request.url.path, method=request.method, status=resp.status_code, latency_ms=dur)

    if _pool is not None:
        async with _pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO api_request_log(ts, endpoint, method, latency_ms, status, api_key_hash) VALUES($1,$2,$3,$4,$5,$6)",
                datetime.now(timezone.utc),
                request.url.path,
                request.method,
                float(dur),
                int(resp.status_code),
                api_hash,
            )

    return resp


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    auth = request.headers.get("Authorization", "")
    key = auth[:64]
    now = time.time()
    q = _rate[key]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= 100:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
    q.append(now)
    return await call_next(request)


@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=30)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"detail": "Request timeout"})


@app.post("/v1/analyze/patient")
async def analyze_patient(request: PatientAnalysisRequest, background_tasks: BackgroundTasks, api_key: dict = Depends(verify_api_key)) -> AnalysisJobResponse:
    _ = background_tasks
    if api_key.get("role") not in ["clinician", "admin"]:
        raise HTTPException(status_code=403, detail="Forbidden")

    if not request.patient_id:
        raise HTTPException(status_code=400, detail="Invalid patient_id")

    await _validate_patient_and_data(request.patient_id, request.analysis_config)

    task = celery_app.send_task("analyze_patient", args=[request.patient_id, request.analysis_config])
    if _pool is not None:
        async with _pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO analysis_jobs(job_id, patient_id, status, config) VALUES($1,$2,$3,$4) ON CONFLICT(job_id) DO UPDATE SET status=EXCLUDED.status, config=EXCLUDED.config",
                task.id,
                request.patient_id,
                "queued",
                json.dumps(request.analysis_config),
            )
    return AnalysisJobResponse(job_id=task.id, status="queued")


@app.get("/v1/analyze/status/{job_id}")
async def analyze_status(job_id: str, api_key: dict = Depends(verify_api_key)):
    _ = api_key
    result = celery_app.AsyncResult(job_id)
    if _pool is not None:
        async with _pool.acquire() as conn:
            await conn.execute("UPDATE analysis_jobs SET status=$1 WHERE job_id=$2", result.status.lower(), job_id)
    return {"job_id": job_id, "status": result.status.lower()}


@app.get("/v1/analyze/result/{job_id}")
async def analyze_result(job_id: str, api_key: dict = Depends(verify_api_key)):
    _ = api_key
    result = celery_app.AsyncResult(job_id)
    if not result.ready():
        raise HTTPException(status_code=202, detail="Result not ready")
    if _pool is not None:
        async with _pool.acquire() as conn:
            await conn.execute(
                "UPDATE analysis_jobs SET status=$1, result=$2::jsonb WHERE job_id=$3",
                result.status.lower(),
                json.dumps(result.result),
                job_id,
            )
    return {"job_id": job_id, "result": result.result}


@app.post("/v1/simulate/intervention")
async def simulate_intervention(payload: dict, api_key: dict = Depends(verify_api_key)):
    if api_key.get("role") not in ["clinician", "admin"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    return {"status": "accepted", "payload": payload}


@app.get("/v1/patient/{patient_id}/history")
async def patient_history(patient_id: str, api_key: dict = Depends(verify_api_key)):
    if api_key.get("role") == "researcher":
        raise HTTPException(status_code=403, detail="Researchers cannot access patient-level history")
    if _pool is None:
        raise HTTPException(status_code=503, detail="Database pool unavailable")
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT job_id, submitted_at, status, result FROM analysis_jobs WHERE patient_id=$1 ORDER BY submitted_at DESC LIMIT 50",
            patient_id,
        )
    history = [dict(r) for r in rows]
    return {"patient_id": patient_id, "history": history}


@app.get("/health")
async def health():
    return {"status": "ok"}


