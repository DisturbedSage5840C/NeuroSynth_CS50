from __future__ import annotations

from celery import Celery

from backend.core.config import get_settings

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
settings = get_settings()

celery_app = Celery(
    "neurosynth",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["backend.tasks"],
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    worker_prefetch_multiplier=1,
    # v5 periodic tasks
    beat_schedule={
        # Check each data source for staleness and mark pending if > 7 days old.
        # Runs daily at 03:00 UTC — outside peak clinical usage.
        "v5-data-source-refresh-check": {
            "task": "backend.tasks.check_data_source_freshness",
            "schedule": 86400.0,  # every 24 hours
            "options": {"expires": 3600},
        },
        # Recompute cohort statistics cache from real_v5.parquet.
        # Runs weekly on Monday 04:00 UTC.
        "v5-cohort-stats-recompute": {
            "task": "backend.tasks.recompute_cohort_stats",
            "schedule": 604800.0,  # every 7 days
            "options": {"expires": 7200},
        },
    },
)
