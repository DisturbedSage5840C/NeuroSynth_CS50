# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""Prometheus metric definitions for NeuroSynth production monitoring."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info


# -- Application info --
APP_INFO = Info("neurosynth", "NeuroSynth application information")

# -- Inference metrics --
INFERENCE_LATENCY = Histogram(
    "neurosynth_inference_latency_seconds",
    "Time to complete a prediction request",
    labelnames=["endpoint", "model", "api_version"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

INFERENCE_REQUESTS = Counter(
    "neurosynth_inference_requests_total",
    "Total prediction requests",
    labelnames=["endpoint", "status", "api_version"],
)

INFERENCE_ERRORS = Counter(
    "neurosynth_inference_errors_total",
    "Total prediction errors",
    labelnames=["endpoint", "error_type", "api_version"],
)

# -- Model performance --
MODEL_AUC = Gauge(
    "neurosynth_model_auc",
    "Current model AUC-ROC score",
    labelnames=["model_name", "disease"],
)

MODEL_ECE = Gauge(
    "neurosynth_model_ece",
    "Expected Calibration Error",
    labelnames=["model_name"],
)

MODEL_F1 = Gauge(
    "neurosynth_model_f1",
    "Model F1 score (weighted)",
    labelnames=["model_name"],
)

# -- Drift metrics --
DRIFT_PSI = Gauge(
    "neurosynth_drift_psi",
    "Population Stability Index per feature",
    labelnames=["feature"],
)

DRIFT_KS = Gauge(
    "neurosynth_drift_ks_statistic",
    "KS test statistic per feature",
    labelnames=["feature"],
)

DRIFT_SEVERITY = Gauge(
    "neurosynth_drift_severity",
    "Overall drift severity (0=none, 1=minor, 2=warning, 3=critical)",
)

DRIFT_FEATURES_DRIFTED = Gauge(
    "neurosynth_drift_features_drifted",
    "Number of features currently drifted",
)

# -- Validation gates --
GATE_STATUS = Gauge(
    "neurosynth_gate_status",
    "Validation gate status (1=pass, 0=fail)",
    labelnames=["gate_name", "gate_type"],
)

GATE_DECISION = Gauge(
    "neurosynth_gate_decision",
    "Overall gate decision (1=promote, 0=reject, 0.5=review)",
)

# -- Circuit breaker --
CIRCUIT_BREAKER_STATE = Gauge(
    "neurosynth_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open)",
    labelnames=["endpoint"],
)

CIRCUIT_BREAKER_FAILURES = Counter(
    "neurosynth_circuit_breaker_failures_total",
    "Circuit breaker failure count",
    labelnames=["endpoint"],
)

# -- Resource metrics --
GPU_MEMORY = Gauge(
    "neurosynth_gpu_memory_used_bytes",
    "GPU memory utilization",
    labelnames=["device"],
)

MODEL_LOAD_TIME = Histogram(
    "neurosynth_model_load_seconds",
    "Time to load a model into memory",
    labelnames=["model_name"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# -- Data pipeline --
PIPELINE_RECORDS_PROCESSED = Counter(
    "neurosynth_pipeline_records_processed_total",
    "Total records processed by data pipeline",
    labelnames=["stage", "connector"],
)

PIPELINE_ERRORS = Counter(
    "neurosynth_pipeline_errors_total",
    "Pipeline processing errors",
    labelnames=["stage", "error_type"],
)

# -- Report generation --
REPORT_GENERATION_LATENCY = Histogram(
    "neurosynth_report_generation_seconds",
    "Clinical report generation time",
    labelnames=["format"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0],
)


def update_drift_metrics(drift_report) -> None:  # type: ignore[no-untyped-def]
    """Update Prometheus drift metrics from a DriftReport."""
    severity_map = {"NO_DRIFT": 0, "MINOR": 1, "WARNING": 2, "CRITICAL": 3}

    for fr in getattr(drift_report, "feature_results", []):
        DRIFT_PSI.labels(feature=fr.feature).set(fr.psi)
        DRIFT_KS.labels(feature=fr.feature).set(fr.ks_stat)

    overall = str(getattr(drift_report, "overall_severity", "NO_DRIFT"))
    if hasattr(overall, "value"):
        overall = overall.value
    DRIFT_SEVERITY.set(severity_map.get(overall, 0))
    DRIFT_FEATURES_DRIFTED.set(getattr(drift_report, "drifted_features", 0))
