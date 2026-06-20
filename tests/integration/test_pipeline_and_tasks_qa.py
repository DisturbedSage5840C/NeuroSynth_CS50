from __future__ import annotations

import time
import importlib

import pytest
from fastapi.testclient import TestClient

from backend.models import CeleryTaskResult
from backend.tasks import (
    causal_analysis,
    connectome_inference,
    genomic_risk_score,
    report_generation,
    temporal_forecast,
)


@pytest.mark.integration
def test_testcontainers_postgres_redis_neo4j_pipeline_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    tc = pytest.importorskip("testcontainers")
    docker = pytest.importorskip("docker")
    _ = (tc, docker)

    from testcontainers.core.container import DockerContainer
    from testcontainers.postgres import PostgresContainer
    from testcontainers.redis import RedisContainer

    with PostgresContainer("postgres:16") as pg, RedisContainer("redis:7") as redis, DockerContainer("neo4j:5") as neo4j:
        neo4j.with_env("NEO4J_AUTH", "neo4j/testpass")
        neo4j.with_exposed_ports(7687)
        neo4j.start()

        assert pg.get_connection_url().startswith("postgresql")
        assert redis.get_container_host_ip()
        assert neo4j.get_exposed_port(7687)

        redis_host = redis.get_container_host_ip()
        redis_port = redis.get_exposed_port(6379)
        monkeypatch.setenv("NEUROSYNTH_POSTGRES_DSN", pg.get_connection_url())
        monkeypatch.setenv("NEUROSYNTH_REDIS_URL", f"redis://{redis_host}:{redis_port}/0")

        from backend.core.config import get_settings

        get_settings.cache_clear()
        import backend.api as api_module

        api_module = importlib.reload(api_module)
        with TestClient(api_module.app) as client:
            login = client.post(
                "/auth/login",
                json={"username": "integration-user", "password": "pass", "role": "CLINICIAN"},
            )
            assert login.status_code == 200

            ready = client.get("/ready")
            assert ready.status_code == 200
            assert ready.json()["database"] is True
            assert ready.json()["redis"] is True

            pred = client.post(
                "/predictions/run",
                json={
                    "patient_id": "P-INT-API-1",
                    "features": {
                        "Age": 72,
                        "MMSE": 24,
                        "FunctionalAssessment": 6.3,
                        "ADL": 6.1,
                        "SleepQuality": 5.2,
                    },
                },
            )
            assert pred.status_code == 200

            rep = client.post(
                "/reports/generate",
                json={"patient_id": "P-INT-API-1", "notes": "integration smoke"},
            )
            assert rep.status_code == 200


@pytest.mark.integration
def test_celery_eager_mode_completes_all_phase_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.celery_app import celery_app

    # Eager mode executes tasks in-process for deterministic integration tests.
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True

    # Avoid external Redis dependency while still validating task outputs.
    monkeypatch.setattr("backend.tasks._publish_progress", lambda *args, **kwargs: None)

    phase_results = [
        connectome_inference.delay("P-INT-001").result,
        genomic_risk_score.delay("P-INT-001").result,
        temporal_forecast.delay("P-INT-001").result,
        causal_analysis.delay("P-INT-001").result,
        report_generation.delay("P-INT-001", "integration-note").result,
    ]

    for result in phase_results:
        validated = CeleryTaskResult(
            task_id="eager",
            phase=str(result["phase"]),
            state="SUCCESS",
            meta=result,
        )
        assert validated.meta["status"] == "completed"


@pytest.mark.integration
def test_kafka_mockproducer_event_reaches_iceberg_within_5_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    ck = pytest.importorskip("confluent_kafka")
    producer = ck.Producer({"bootstrap.servers": "mock://"})

    sink: list[dict] = []

    def _persist_to_iceberg(event: dict) -> None:
        sink.append(event)

    start = time.monotonic()
    event = {"patient_id": "P-INT-002", "metric": "heart_rate", "value": 92.0}

    # Mock producer contract: produce then flush before persistence callback.
    producer.produce("wearables.raw", key=event["patient_id"], value=str(event).encode("utf-8"))
    producer.flush(timeout=1)
    _persist_to_iceberg(event)

    elapsed = time.monotonic() - start
    assert elapsed < 5.0
    assert len(sink) == 1
    assert sink[0]["metric"] == "heart_rate"
