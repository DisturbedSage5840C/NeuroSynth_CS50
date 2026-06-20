"""Priority 8 verification tests — Monitoring & Drift Detection."""
import json
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import numpy as np

print("=" * 60)
print("NeuroSynth v2 — Priority 8 Monitoring Verification")
print("=" * 60)

# 1. Drift Detector — PSI computation
print("\n[1/6] Drift Detector — PSI + KS computation...")
from neurosynth.monitoring.drift_detector import DriftDetector, DriftSeverity

detector = DriftDetector()

# No drift scenario: split one dataset in half
rng = np.random.RandomState(42)
full = rng.normal(50, 10, (1000, 5))
ref = full[:500]
cur = full[500:]
features = ["MMSE", "Age", "BMI", "SleepQuality", "PhysicalActivity"]
report_no = detector.detect(ref, cur, features)
print(f"  No-drift test: severity={report_no.overall_severity.value}, drifted={report_no.drifted_features}")
assert report_no.overall_severity in (DriftSeverity.NO_DRIFT, DriftSeverity.MINOR), f"Expected no/minor drift, got {report_no.overall_severity}"
print("  ✅ PASSED")

# 2. Drift Detector — Critical drift
print("\n[2/6] Drift Detector — Critical drift detection...")
ref_drift = rng.normal(50, 10, (500, 3))
cur_drift = rng.normal(80, 5, (500, 3))  # Massively shifted
report_crit = detector.detect(ref_drift, cur_drift, ["MMSE", "Age", "BMI"])
print(f"  Critical test: severity={report_crit.overall_severity.value}, drifted={report_crit.drifted_features}")
for fr in report_crit.feature_results[:3]:
    print(f"    {fr.feature}: PSI={fr.psi:.4f} KS={fr.ks_stat:.4f} → {fr.severity.value}")
assert report_crit.overall_severity == DriftSeverity.CRITICAL
assert report_crit.drifted_features > 0
print("  ✅ PASSED")

# 3. Drift Report serialization
print("\n[3/6] DriftReport serialization...")
report_dict = report_crit.to_dict()
assert "features" in report_dict
assert "overall_severity" in report_dict
assert report_dict["overall_severity"] == "CRITICAL"
json_str = json.dumps(report_dict, indent=2)
print(f"  JSON size: {len(json_str)} chars")
print(f"  Severity: {report_dict['overall_severity']}")
print(f"  Recommendation: {report_dict['recommendation'][:60]}...")
print("  ✅ PASSED")

# 4. Alerting — Alert creation from drift report
print("\n[4/6] Alerting — drift-to-alert conversion...")
from neurosynth.monitoring.alerting import (
    AlertDispatcher, Alert, AlertPriority, AlertChannel, create_drift_alert
)

alert = create_drift_alert(report_crit)
assert alert.priority == AlertPriority.CRITICAL
assert "CRITICAL" in alert.title
assert alert.metadata["drifted_features"] > 0
print(f"  Title: {alert.title}")
print(f"  Priority: {alert.priority.value}")
print(f"  Source: {alert.source}")
print(f"  Metadata: {alert.metadata}")
print("  ✅ PASSED")

# 5. Alerting — log dispatch
print("\n[5/6] Alerting — log channel dispatch...")
dispatcher = AlertDispatcher(channels=[AlertChannel.LOG])
results = dispatcher.dispatch(alert)
assert results.get("log") is True
print(f"  Channels dispatched: {results}")
print("  ✅ PASSED")

# 6. Prometheus metrics
print("\n[6/6] Prometheus metric definitions...")
from neurosynth.monitoring.metrics import (
    INFERENCE_LATENCY, INFERENCE_REQUESTS, INFERENCE_ERRORS,
    MODEL_AUC, MODEL_ECE, DRIFT_PSI, DRIFT_KS, DRIFT_SEVERITY,
    GATE_STATUS, CIRCUIT_BREAKER_STATE, update_drift_metrics,
)

# Verify metrics exist and can be set
MODEL_AUC.labels(model_name="ensemble", disease="Alzheimer's").set(0.819)
MODEL_ECE.labels(model_name="ensemble").set(0.020)
DRIFT_PSI.labels(feature="MMSE").set(0.05)
GATE_STATUS.labels(gate_name="auc_threshold", gate_type="hard").set(1)
CIRCUIT_BREAKER_STATE.labels(endpoint="/v2/predictions/analyze").set(0)

# Test update_drift_metrics helper
update_drift_metrics(report_crit)
print(f"  Metrics registered: INFERENCE_LATENCY, INFERENCE_REQUESTS, MODEL_AUC, DRIFT_PSI, etc.")
print(f"  update_drift_metrics: updated {len(report_crit.feature_results)} feature gauges")
print("  ✅ PASSED")

# BONUS: Infrastructure files
print("\n[BONUS] Infrastructure files check...")
prom_path = os.path.join(os.path.dirname(__file__), "..", "infrastructure", "prometheus", "prometheus.yml")
graf_path = os.path.join(os.path.dirname(__file__), "..", "infrastructure", "grafana", "dashboards", "neurosynth.json")
assert os.path.exists(prom_path), f"Missing: {prom_path}"
assert os.path.exists(graf_path), f"Missing: {graf_path}"
with open(graf_path) as f:
    dashboard = json.load(f)
    panel_count = len(dashboard.get("panels", []))
print(f"  Prometheus config: ✅")
print(f"  Grafana dashboard: ✅ ({panel_count} panels)")
print("  ✅ PASSED")

print("\n" + "=" * 60)
print("ALL PRIORITY 8 VERIFICATION TESTS PASSED")
print("=" * 60)
