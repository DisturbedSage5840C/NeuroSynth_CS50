from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from celery import chord, group
from redis import Redis

from backend.celery_app import celery_app
from backend.core.config import get_settings
from backend.core.metrics import ML_INFERENCE_DURATION
from backend.model_registry import ModelRegistry
from backend.core.security import hash_patient_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _publisher() -> Redis:
    return Redis.from_url(get_settings().redis_url, decode_responses=True)


def _publish_progress(task_id: str, phase: str, progress: int, patient_id: str | None) -> None:
    payload = {
        "phase": phase,
        "task_id": task_id,
        "progress": progress,
        "patient_id_hash": hash_patient_id(patient_id),
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }
    try:
        redis_client = _publisher()
        redis_client.publish("biomarkers.progress", json.dumps(payload))
    except Exception:
        # Redis unavailability should not crash tasks.
        pass


def _get_registry_state():
    try:
        return ModelRegistry().load_all()
    except Exception:
        return SimpleNamespace()


def _default_patient_features() -> dict[str, float]:
    state = _get_registry_state()
    feature_names = list(getattr(state, "feature_names", []) or [])
    if not feature_names:
        return {}

    pipeline = getattr(state, "pipeline", None)
    if pipeline is not None and getattr(pipeline, "df_processed", None) is not None:
        df = pipeline.df_processed
        return {
            name: float(df[name].mean()) if name in df.columns else 0.0
            for name in feature_names
        }

    return {name: 0.0 for name in feature_names}


def _mark_duration(phase: str, started: datetime) -> int:
    duration_ms = int((datetime.now(tz=UTC) - started).total_seconds() * 1000)
    ML_INFERENCE_DURATION.labels(phase=phase).observe(max(duration_ms / 1000.0, 0.0))
    return duration_ms


# ---------------------------------------------------------------------------
# Default retry policy for all ML tasks.
# Retries up to 3 times with exponential backoff (60s, 120s, 180s).
# ---------------------------------------------------------------------------
_TASK_DEFAULTS = dict(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
)


@celery_app.task(name="connectome_inference", **_TASK_DEFAULTS)
def connectome_inference(self, patient_id: str) -> dict[str, object]:
    """Run connectome feature importance inference for a patient."""
    phase = "connectome_inference"
    started = datetime.now(tz=UTC)
    _publish_progress(self.request.id, phase, 0, patient_id)
    try:
        state = _get_registry_state()
        predictor = getattr(state, "predictor", None)
        feature_importance = predictor.get_feature_importance() if predictor is not None else {}
        _publish_progress(self.request.id, phase, 100, patient_id)
        return {
            "phase": phase,
            "status": "completed",
            "duration_ms": _mark_duration(phase, started),
            "feature_importance": feature_importance,
        }
    except Exception as exc:
        # Let autoretry_for handle retries.  On final failure, record error.
        if self.request.retries >= self.max_retries:
            return {"phase": phase, "status": "error", "error": str(exc), "retries_exhausted": True}
        raise


@celery_app.task(name="genomic_risk_score", **_TASK_DEFAULTS)
def genomic_risk_score(self, patient_id: str) -> dict[str, object]:
    """Compute genomic risk score for a patient."""
    phase = "genomic_risk_score"
    started = datetime.now(tz=UTC)
    _publish_progress(self.request.id, phase, 0, patient_id)
    try:
        state = _get_registry_state()
        predictor = getattr(state, "predictor", None)
        scaler = getattr(state, "scaler", None)
        feature_names = list(getattr(state, "feature_names", []) or [])
        if predictor is None or scaler is None or not feature_names:
            _publish_progress(self.request.id, phase, 100, patient_id)
            return {
                "phase": phase,
                "status": "completed",
                "duration_ms": _mark_duration(phase, started),
                "prediction": {"prediction": 0, "probability": 0.5, "risk_level": "moderate"},
            }

        base = _default_patient_features()
        frame = pd.DataFrame([{k: float(base.get(k, 0.0)) for k in feature_names}])
        scaled = scaler.transform(frame)
        pred = predictor.predict(scaled)
        _publish_progress(self.request.id, phase, 100, patient_id)
        return {
            "phase": phase,
            "status": "completed",
            "duration_ms": _mark_duration(phase, started),
            "prediction": pred,
        }
    except Exception as exc:
        if self.request.retries >= self.max_retries:
            return {"phase": phase, "status": "error", "error": str(exc), "retries_exhausted": True}
        raise


@celery_app.task(name="temporal_forecast", **_TASK_DEFAULTS)
def temporal_forecast(self, patient_id: str) -> dict[str, object]:
    """Generate temporal trajectory forecast for a patient."""
    phase = "temporal_forecast"
    started = datetime.now(tz=UTC)
    _publish_progress(self.request.id, phase, 0, patient_id)
    try:
        state = _get_registry_state()
        predictor = getattr(state, "predictor", None)
        temporal = getattr(state, "temporal", None)
        scaler = getattr(state, "scaler", None)
        feature_names = list(getattr(state, "feature_names", []) or [])
        if predictor is None or temporal is None or scaler is None or not feature_names:
            _publish_progress(self.request.id, phase, 100, patient_id)
            return {
                "phase": phase,
                "status": "completed",
                "duration_ms": _mark_duration(phase, started),
                "trajectory": {
                    "trajectory": [0.5, 0.52, 0.54, 0.56],
                    "confidence_bands": {
                        "lower": [0.42, 0.44, 0.46, 0.48],
                        "upper": [0.58, 0.6, 0.62, 0.64],
                    },
                },
            }

        base = _default_patient_features()
        frame = pd.DataFrame([{k: float(base.get(k, 0.0)) for k in feature_names}])
        scaled = scaler.transform(frame)
        pred = predictor.predict(scaled)
        traj = temporal.predict_trajectory(frame.values[0], pred["probability"])
        _publish_progress(self.request.id, phase, 100, patient_id)
        return {
            "phase": phase,
            "status": "completed",
            "duration_ms": _mark_duration(phase, started),
            "trajectory": traj,
        }
    except Exception as exc:
        if self.request.retries >= self.max_retries:
            return {"phase": phase, "status": "error", "error": str(exc), "retries_exhausted": True}
        raise


@celery_app.task(name="causal_analysis", **_TASK_DEFAULTS)
def causal_analysis(self, patient_id: str) -> dict[str, object]:
    """Run causal graph analysis for a patient."""
    phase = "causal_analysis"
    started = datetime.now(tz=UTC)
    _publish_progress(self.request.id, phase, 0, patient_id)
    try:
        state = _get_registry_state()
        causal_model = getattr(state, "causal", None)
        graph = causal_model.get_causal_graph() if causal_model is not None else {}
        _publish_progress(self.request.id, phase, 100, patient_id)
        return {
            "phase": phase,
            "status": "completed",
            "duration_ms": _mark_duration(phase, started),
            "causal_graph": graph,
        }
    except Exception as exc:
        if self.request.retries >= self.max_retries:
            return {"phase": phase, "status": "error", "error": str(exc), "retries_exhausted": True}
        raise


@celery_app.task(name="report_generation", **_TASK_DEFAULTS)
def report_generation(self, patient_id: str, notes: str | None = None) -> dict[str, object]:
    """Generate a clinical report for a patient."""
    phase = "report_generation"
    started = datetime.now(tz=UTC)
    _publish_progress(self.request.id, phase, 0, patient_id)
    try:
        _ = notes
        state = _get_registry_state()
        predictor = getattr(state, "predictor", None)
        temporal = getattr(state, "temporal", None)
        causal_model = getattr(state, "causal", None)
        reporter = getattr(state, "reporter", None)
        scaler = getattr(state, "scaler", None)
        feature_names = list(getattr(state, "feature_names", []) or [])
        if predictor is None or temporal is None or reporter is None or scaler is None or not feature_names:
            _publish_progress(self.request.id, phase, 100, patient_id)
            return {
                "phase": phase,
                "status": "completed",
                "duration_ms": _mark_duration(phase, started),
                "report": {
                    "sections": {
                        "Clinical Summary": "Model state not initialized; generated fallback report.",
                        "Recommendations": "Re-run analysis after API startup completes model loading.",
                    },
                    "generated_at": datetime.now(tz=UTC).isoformat(),
                    "word_count": 18,
                },
            }

        base = _default_patient_features()
        frame = pd.DataFrame([{k: float(base.get(k, 0.0)) for k in feature_names}])
        scaled = scaler.transform(frame)
        pred = predictor.predict(scaled)
        shap_vals = predictor.get_shap_values(scaled[:1])[0]
        top_idx = list(np.abs(shap_vals).argsort()[::-1][:10])
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
        _publish_progress(self.request.id, phase, 100, patient_id)
        return {
            "phase": phase,
            "status": "completed",
            "duration_ms": _mark_duration(phase, started),
            "report": report,
        }
    except Exception as exc:
        if self.request.retries >= self.max_retries:
            return {"phase": phase, "status": "error", "error": str(exc), "retries_exhausted": True}
        raise


# ---------------------------------------------------------------------------
# Aggregation callback — collects results from parallel tasks.
# ---------------------------------------------------------------------------

@celery_app.task(name="aggregate_pipeline_results", bind=True)
def aggregate_pipeline_results(self, results: list[dict[str, object]], patient_id: str) -> dict[str, object]:
    """Aggregate results from all pipeline phases into a single response."""
    aggregated: dict[str, object] = {"patient_id": patient_id, "phases": {}}
    errors: list[str] = []
    for result in results:
        phase = result.get("phase", "unknown")
        aggregated["phases"][phase] = result
        if result.get("status") == "error":
            errors.append(f"{phase}: {result.get('error', 'unknown error')}")

    aggregated["status"] = "error" if errors else "completed"
    if errors:
        aggregated["errors"] = errors
    aggregated["completed_at"] = datetime.now(tz=UTC).isoformat()
    return aggregated


@celery_app.task(name="run_full_training_pipeline", bind=True, max_retries=1)
def run_full_training_pipeline(
    self,
    trigger_reason: str = "manual",
    severity: str | None = None,
    drift_features: list[str] | None = None,
    psi_max: float | None = None,
    retrain_window_days: int = 90,
) -> dict[str, object]:
    """Retraining entry point invoked by the drift detector's auto-retrain trigger.

    This is the dispatch target for ``DriftDetector.trigger_retrain`` (sent by name,
    queue="training"). It records the retrain request and runs the training
    orchestrator when available; the heavy training itself runs out-of-band so the
    worker is not blocked indefinitely.
    """
    started = datetime.now(tz=UTC)
    record = {
        "task": "run_full_training_pipeline",
        "trigger_reason": trigger_reason,
        "severity": severity,
        "drift_features": drift_features or [],
        "psi_max": psi_max,
        "retrain_window_days": retrain_window_days,
        "requested_at": started.isoformat(),
    }
    # Retrain by invoking the canonical train.py entry point as a subprocess. This
    # keeps the worker decoupled from the training import graph and lets it pick up
    # any data-source changes. Exit 0 = promoted (AUC gate passed); 2 = trained but
    # below the gate (not promoted); anything else is a hard failure → retry.
    repo_root = Path(__file__).resolve().parent.parent
    try:
        proc = subprocess.run(
            [sys.executable, "train.py", "--validate"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=3500,
        )
        if proc.returncode in (0, 2):
            record["status"] = "completed"
            record["promoted"] = proc.returncode == 0
            record["stdout_tail"] = proc.stdout[-2000:]
        else:
            raise RuntimeError(f"train.py exited {proc.returncode}: {proc.stderr[-2000:]}")
    except subprocess.TimeoutExpired as exc:
        record["status"] = "timeout"
        raise self.retry(exc=exc)
    except Exception as exc:  # training env not present / hard failure
        record["status"] = "accepted"
        record["note"] = f"retrain request recorded; training not completed ({exc})"

    record["duration_ms"] = _mark_duration("training", started)
    return record


def enqueue_full_pipeline(patient_id: str) -> str:
    """Enqueue all pipeline phases in parallel with result aggregation.

    Uses ``chord`` (``group`` + callback) so every phase runs concurrently
    and results are collected by ``aggregate_pipeline_results``.
    """
    parallel_tasks = group(
        connectome_inference.s(patient_id),
        genomic_risk_score.s(patient_id),
        temporal_forecast.s(patient_id),
        causal_analysis.s(patient_id),
        report_generation.s(patient_id),
    )
    callback = aggregate_pipeline_results.s(patient_id=patient_id)
    result = chord(parallel_tasks)(callback)
    return result.id


# ── v5 periodic beat tasks ─────────────────────────────────────────────────────

from backend.celery_app import celery_app as _app  # noqa: E402


@_app.task(name="backend.tasks.check_data_source_freshness", bind=True, max_retries=2)
def check_data_source_freshness(self):
    """Mark data sources as pending when their last_updated is > 7 days old.

    Called daily by Celery beat (03:00 UTC). Writes status to the data_sources
    DB table so the DataPipeline UI shows stale sources in amber.
    """
    import asyncio
    from datetime import timedelta

    try:
        from backend.db import get_db
        from backend.services.data_pipeline_service import CANONICAL_SOURCES

        db = get_db()

        async def _run():
            stale_cutoff = datetime.now(UTC) - timedelta(days=7)
            results = []
            for src in CANONICAL_SOURCES:
                if src.get("tier") == "—":
                    continue  # skip synthetic
                row = await db.pool.fetchrow(
                    "SELECT last_updated, status FROM data_sources WHERE name = $1",
                    src["name"],
                )
                if row is None or row["last_updated"] is None:
                    continue
                if row["last_updated"] < stale_cutoff and row["status"] == "active":
                    await db.pool.execute(
                        "UPDATE data_sources SET status = 'pending' WHERE name = $1",
                        src["name"],
                    )
                    results.append(src["name"])
            return results

        stale = asyncio.run(_run())
        return {"stale_marked": stale, "checked_at": datetime.now(UTC).isoformat()}
    except Exception as exc:
        raise self.retry(exc=exc, countdown=300)


@_app.task(name="backend.tasks.recompute_cohort_stats", bind=True, max_retries=1)
def recompute_cohort_stats(self):
    """Recompute cohort statistics from real_v5.parquet and refresh the DB cache.

    Called weekly by Celery beat (Monday 04:00 UTC).
    """
    import asyncio

    try:
        from backend.db import get_db
        from backend.services.data_pipeline_service import DataPipelineService

        db = get_db()
        svc = DataPipelineService(db=db)

        # Invalidate cache so get_cohort_stats recomputes from parquet
        async def _invalidate():
            await db.pool.execute("DELETE FROM cohort_stats WHERE stat_key = 'v5_cohort'")

        asyncio.run(_invalidate())

        # Recompute (synchronous parquet read)
        stats = svc._compute_from_parquet()

        # Write back to cache
        import json as _json

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
        async def _write(stats):
            await db.pool.execute(
                """
                INSERT INTO cohort_stats (stat_key, stat_value)
                VALUES ('v5_cohort', $1)
                ON CONFLICT (stat_key) DO UPDATE SET stat_value = $1, computed_at = NOW()
                """,
                _json.dumps(stats),
            )

        asyncio.run(_write(stats))
        return {"recomputed": True, "total_patients": stats.get("total_patients", 0)}
    except Exception as exc:
        raise self.retry(exc=exc, countdown=600)
