from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
REQUEST_COUNT = Counter(
    "neurosynth_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

REQUEST_LATENCY = Histogram(
    "neurosynth_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
)

ML_INFERENCE_DURATION = Histogram(
    "neurosynth_ml_inference_duration_seconds",
    "Duration of ML phase inferences",
    ["phase"],
)

ACTIVE_SSE_CONNECTIONS = Gauge(
    "neurosynth_active_sse_connections",
    "Number of active biomarker SSE clients",
)

CELERY_QUEUE_DEPTH = Gauge(
    "neurosynth_celery_queue_depth",
    "Number of queued Celery jobs",
    ["queue"],
)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
