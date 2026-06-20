from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os as _os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from redis.asyncio import Redis
from slowapi.errors import RateLimitExceeded

from backend.celery_app import celery_app
from backend.core.config import get_settings
from backend.core.logging import configure_structlog, get_logger
from backend.core.metrics import CELERY_QUEUE_DEPTH, REQUEST_COUNT, REQUEST_LATENCY, render_metrics
from backend.core.rate_limit import limiter, rate_limit_exceeded_handler
from backend.core.security import ACCESS_COOKIE, Role, decode_token, hash_patient_id
from backend.db import get_db
from backend.deps import require_role
from backend.routers import admin, auth, biomarkers, causal, data as data_router, features, health, literature, patients, pipelines, predictions, predictions_v2, predictions_v3, reports, reports_v2


def _dataset_path() -> Path:
    # Must match backend.data_pipeline.DataPipeline.DATA_CANDIDATES so the manifest
    # MD5 check validates against the same dataset training wrote artifacts from.
    for candidate in (
        "data/realistic_v4.parquet",
        "data/realistic_v4.csv",
        "neurological_disease_data.csv",
        "alzheimers_disease_data.csv",
    ):
        if Path(candidate).exists():
            return Path(candidate)
    return Path("alzheimers_disease_data.csv")


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_valid(models_dir: Path, dataset_file: Path) -> bool:
    """Return True when pre-trained artifacts are present and consistent.

    Dataset presence/MD5 is only checked when the file actually exists —
    cloud deploys ship model artifacts without training data, which is fine.
    """
    _log = logging.getLogger("neurosynth.bootstrap")
    manifest_file = models_dir / "model_manifest.json"
    if not manifest_file.exists():
        _log.warning("manifest_check_failed: manifest file missing at %s", manifest_file)
        return False

    # Core artifacts required for inference (v5 path; rf_model.pkl only needed
    # as legacy fallback when ensemble_v5/ is absent).
    required = [
        "scaler.pkl",
        "gb_model.pkl",
        "lr_model.pkl",
        "causal_graph.npy",
        "disease_clf.pkl",
        "disease_le.pkl",
        "disease_features.pkl",
    ]
    for file_name in required:
        if not (models_dir / file_name).exists():
            _log.warning("manifest_check_failed: required artifact missing: %s", file_name)
            return False

    # Dataset MD5 check only when training data is present (local dev / CI).
    if dataset_file.exists():
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception as e:
            _log.warning("manifest_check_failed: corrupt manifest JSON: %s", e)
            return False
        expected_md5 = str(manifest.get("dataset_md5", ""))
        actual_md5 = _md5(dataset_file)
        if expected_md5 and expected_md5 != actual_md5:
            _log.warning(
                "manifest_check_failed: dataset MD5 mismatch (expected=%s, actual=%s)",
                expected_md5, actual_md5,
            )
            return False

    return True


_pretrain_executor = ThreadPoolExecutor(max_workers=1)


def _run_pretrain_sync(dataset_file: Path, models_dir: Path) -> None:
    """Synchronous pretrain — called inside a thread executor."""
    cmd = [
        sys.executable,
        "scripts/pretrain.py",
        "--dataset",
        str(dataset_file),
        "--models-dir",
        str(models_dir),
    ]
    subprocess.run(cmd, check=True)


async def _run_pretrain(dataset_file: Path, models_dir: Path) -> None:
    """Run pretrain in a thread to avoid blocking the async event loop."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_pretrain_executor, _run_pretrain_sync, dataset_file, models_dir)


async def _drain_celery_queue(timeout_seconds: int = 20) -> None:
    started = time.monotonic()
    while time.monotonic() - started < timeout_seconds:
        try:
            inspect = celery_app.control.inspect(timeout=1.0)
            active = inspect.active() or {}
            reserved = inspect.reserved() or {}
        except Exception:
            # If broker is unavailable during shutdown, exit drain gracefully.
            return
        in_progress = sum(len(v) for v in active.values()) + sum(len(v) for v in reserved.values())
        if in_progress == 0:
            return
        await asyncio.sleep(1.0)


def _pull_models_from_r2(models_dir: Path, logger) -> None:
    """Download trained model artifacts from Cloudflare R2 when models/ is empty.

    No-op unless ``R2_ACCOUNT_ID`` is set, so it's safe locally and in CI. Lets a
    fresh free-tier container (Render) fetch the gitignored weights on startup.
    """
    import os

    if (models_dir / "model_manifest.json").exists():
        return
    account = os.getenv("R2_ACCOUNT_ID")
    if not account:
        return
    try:
        import boto3
        from botocore.config import Config

        s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )
        bucket = os.environ.get("R2_BUCKET", "neurosynth-models")
        paginator = s3.get_paginator("list_objects_v2")
        n = 0
        for page in paginator.paginate(Bucket=bucket, Prefix="models/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                dest = Path(key)
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(dest))
                n += 1
        logger.info("r2_model_artifacts_downloaded", files=n)
    except Exception as exc:
        logger.warning("r2_download_failed", error=str(exc))


async def _keepalive_ping(logger) -> None:
    """Ping /health every 10 min to keep a free-tier (Render) container warm.

    No-op-friendly: only meaningful when KEEPALIVE_ENABLED is set (do not run it
    locally / in tests). Failures are swallowed.
    """
    import os

    if os.getenv("KEEPALIVE_ENABLED", "").lower() not in ("1", "true", "yes"):
        return
    await asyncio.sleep(90)  # let startup finish
    port = os.getenv("PORT", "8000")
    while True:
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                await client.get(f"http://localhost:{port}/health", timeout=5)
        except Exception:
            pass
        await asyncio.sleep(600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os as _os_li
    if _os_li.getenv("TESTING"):
        # Fast path for the test suite — skip all network I/O, heavy ML imports,
        # and subprocess launches.  Tests that need specific app.state values
        # inject them directly after TestClient.__enter__() returns.
        for _attr in ("predictor", "multi_predictor", "temporal", "causal",
                      "disease_classifier", "reporter", "rag", "fusion",
                      "data_pipeline_svc", "scaler", "pipeline"):
            setattr(app.state, _attr, None)
        app.state.feature_names = []
        app.state.dataset_stats = {}
        app.state.metrics = {}
        app.state.models_loaded = False
        app.state.redis = None
        _keepalive = asyncio.create_task(asyncio.sleep(0))
        app.state.keepalive_task = _keepalive
        yield
        _keepalive.cancel()
        return

    settings = get_settings()
    configure_structlog()
    logger = get_logger("neurosynth.bootstrap")

    # Pre-initialise all state to safe defaults so every endpoint works
    # immediately after yield, even before model loading completes.
    for _attr in ("predictor", "multi_predictor", "temporal", "causal",
                  "disease_classifier", "reporter", "rag", "fusion",
                  "data_pipeline_svc", "scaler", "pipeline"):
        setattr(app.state, _attr, None)
    app.state.feature_names = []
    app.state.dataset_stats = {}
    app.state.metrics = {}
    app.state.models_loaded = False
    app.state.startup_error = ""
    app.state.redis = None

    db = get_db()
    try:
        # asyncpg.create_pool establishes min_size connections synchronously
        # before returning; without an OS-level timeout the SSL/SCRAM handshake
        # to Neon can hang for 2+ minutes and prevent lifespan from yielding.
        await asyncio.wait_for(db.connect(), timeout=15)
        logger.info("database_connected")
        # Apply idempotent schema so fresh deploys (Neon/Render) have tables.
        try:
            from pathlib import Path as _Path

            schema_path = _Path(__file__).with_name("db_schema.sql")
            if schema_path.exists():
                await asyncio.wait_for(
                    db.apply_schema(schema_path.read_text(encoding="utf-8")),
                    timeout=20,
                )
                logger.info("database_schema_applied")
        except Exception as schema_exc:
            logger.warning("database_schema_apply_failed", error=str(schema_exc))
    except Exception as exc:
        logger.warning("database_connect_failed", error=str(exc))

    try:
        app.state.redis = Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        await asyncio.wait_for(app.state.redis.ping(), timeout=8)
        logger.info("redis_connected")
    except Exception as exc:
        app.state.redis = None
        logger.warning("redis_connect_failed", error=str(exc))

    # ── Background model loader ───────────────────────────────────────────────
    # Model loading (CatBoost + LightGBM + RF + others) can take 60-120 s on a
    # cold Render free-tier container.  Running it before yield blocks uvicorn
    # from accepting any connections, so Render's 60 s health-check times out
    # and marks the deploy update_failed.  By loading after yield the server
    # starts immediately; /health passes; /ready shows models_loaded:false until
    # the task completes, then flips to true.
    async def _bg_load_models() -> None:
        try:
            from backend.model_registry import ModelRegistry
            from backend.report_generator_v4 import ClinicalReportGeneratorV4

            models_dir = Path("models")
            _loop = asyncio.get_event_loop()

            # R2 pull is synchronous — run in thread so event loop stays free.
            await _loop.run_in_executor(None, _pull_models_from_r2, models_dir, logger)

            dataset_file = _dataset_path()
            # Skip pretrain when pre-built artifacts are present (cloud deploy)
            # or when SKIP_PRETRAIN env var is set.
            v5_present = (models_dir / "ensemble_v5" / "model_manifest_v5.json").exists()
            from_cache = _manifest_valid(models_dir, dataset_file)
            if not from_cache and not v5_present and not _os.getenv("SKIP_PRETRAIN"):
                await _run_pretrain(dataset_file=dataset_file, models_dir=models_dir)

            # joblib.load calls are CPU/IO-bound — offload to thread pool.
            registry = await _loop.run_in_executor(
                None, lambda: ModelRegistry(models_dir=models_dir).load_all()
            )

            reporter = ClinicalReportGeneratorV4(db=db)

            try:
                from src.neurosynth.llm.rag_v2 import PubMedRAG
                app.state.rag = PubMedRAG(db=db, openai_api_key=_os.getenv("OPENAI_API_KEY"))
                logger.info("rag_initialized rag_enabled=%s", app.state.rag.enabled)
            except Exception as _rag_exc:
                app.state.rag = None
                logger.warning("rag_init_failed error=%s", _rag_exc)

            app.state.pipeline = None
            app.state.predictor = registry.predictor
            app.state.multi_predictor = registry.multi_predictor
            app.state.temporal = registry.temporal
            app.state.causal = registry.causal
            app.state.disease_classifier = registry.disease_classifier
            app.state.reporter = reporter
            app.state.scaler = registry.scaler
            app.state.feature_names = registry.feature_names
            app.state.dataset_stats = registry.dataset_stats
            app.state.metrics = registry.manifest.get("metrics", {})
            app.state.models_loaded = True
            logger.info("ml_models_loaded", source="cache" if from_cache else "pretrain")

            try:
                from src.neurosynth.models.fusion import CrossAttentionFusion
                import torch as _torch
                fusion_path = models_dir / "ensemble_v5" / "cross_attention_fusion.pt"
                if fusion_path.exists():
                    fusion = CrossAttentionFusion(n_modalities=5)
                    fusion.load_state_dict(_torch.load(fusion_path, map_location="cpu"))
                    fusion.eval()
                    app.state.fusion = fusion
                    logger.info("cross_attention_fusion_loaded path=%s", fusion_path)
                else:
                    app.state.fusion = None
            except Exception as _fx_exc:
                app.state.fusion = None
                logger.debug("cross_attention_fusion_not_loaded reason=%s", _fx_exc)

            try:
                from backend.services.data_pipeline_service import DataPipelineService
                svc = DataPipelineService(db=db)
                await svc.upsert_sources()
                app.state.data_pipeline_svc = svc
                logger.info("data_pipeline_service_initialized")
            except Exception as _svc_exc:
                app.state.data_pipeline_svc = None
                logger.warning("data_pipeline_service_failed error=%s", _svc_exc)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            import traceback as _tb
# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
            err_detail = f"{exc}\n{_tb.format_exc()}"
            logger.warning("ml_models_load_failed", error=err_detail)
            app.state.models_loaded = False
            app.state.startup_error = err_detail

    model_task = asyncio.create_task(_bg_load_models())
    app.state.model_task = model_task

    # Fire-and-forget keepalive (only active when KEEPALIVE_ENABLED is set).
    app.state.keepalive_task = asyncio.create_task(_keepalive_ping(logger))

    yield  # ← server starts accepting connections here; health check passes immediately

    keepalive_task = getattr(app.state, "keepalive_task", None)
    if keepalive_task is not None:
        keepalive_task.cancel()
    model_task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(model_task), timeout=2.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass
    await _drain_celery_queue()
    redis_client = getattr(app.state, "redis", None)
    if redis_client is not None:
        await redis_client.close()
    await db.disconnect()


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production API for NeuroSynth healthcare AI workflows with async orchestration and role-based access.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_origin_regex=r"http://localhost:\d+" if settings.app_env == "dev" else None,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


_DEV_BYPASS_AUTH = _os.getenv("DEV_BYPASS_AUTH", "").lower() in ("1", "true", "yes")

@app.middleware("http")
async def auth_context_middleware(request: Request, call_next):
    # DEV_BYPASS_AUTH=true — inject a dev superuser so all endpoints work locally
    # without a running database. Never set this in production.
    if _DEV_BYPASS_AUTH:
        request.state.user = {"user_id": "dev-local", "role": "ADMIN"}
        return await call_next(request)

    # Accept token from cookie OR Authorization: Bearer header
    token = request.cookies.get(ACCESS_COOKIE)
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

    request.state.user = None
    if token:
        try:
            payload = decode_token(token, expected_type="access")
            request.state.user = {
                "user_id": str(payload["sub"]),
                "role": str(payload["role"]),
            }
        except Exception:
            request.state.user = None
    return await call_next(request)


@app.middleware("http")
async def telemetry_middleware(request: Request, call_next):
    logger = get_logger("neurosynth.request")
    trace_id = str(uuid4())
    started = time.perf_counter()

    patient_id = request.headers.get("x-patient-id")
    patient_id_hash = hash_patient_id(patient_id)
    response = await call_next(request)
    latency_s = time.perf_counter() - started
    user = getattr(request.state, "user", None) or {"role": "ANON"}
    path = request.url.path
    REQUEST_COUNT.labels(method=request.method, path=path, status=str(response.status_code)).inc()
    REQUEST_LATENCY.labels(method=request.method, path=path).observe(latency_s)

    logger.info(
        "request_completed",
        trace_id=trace_id,
        role=user.get("role"),
        patient_id=patient_id_hash,
        latency_ms=round(latency_s * 1000, 2),
        method=request.method,
        path=path,
        status_code=response.status_code,
    )
    response.headers["X-Trace-Id"] = trace_id
    return response


app.include_router(health.router)
app.include_router(auth.router)
app.include_router(patients.router)
app.include_router(predictions.router)
app.include_router(reports.router)
app.include_router(causal.router)
app.include_router(biomarkers.router)
app.include_router(admin.router)
app.include_router(pipelines.router)
# v2 routers
app.include_router(predictions_v2.router)
app.include_router(reports_v2.router)
app.include_router(features.router)
# v3 routers
app.include_router(literature.router)
app.include_router(data_router.router)
app.include_router(predictions_v3.router)


@app.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Admin-only metrics endpoint for scraping by Prometheus.",
)
async def metrics_root(_: object = Depends(require_role(Role.ADMIN))) -> Response:
    redis_client = getattr(app.state, "redis", None)
    if redis_client is not None:
        depth = int(await redis_client.llen("celery"))
        CELERY_QUEUE_DEPTH.labels(queue="celery").set(depth)
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)


static_dir = Path("static")
if static_dir.exists():
    assets_dir = static_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

    @app.get("/", include_in_schema=False)
    async def frontend_index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def frontend_spa_fallback(full_path: str) -> FileResponse:
        candidate = static_dir / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(static_dir / "index.html")
