from __future__ import annotations

import json
import random
import time
from typing import Any

from locust import HttpUser, between, events, task

# CI gate threshold (seconds): fail if p95 is above this value.
P95_FAIL_THRESHOLD_SECONDS = 3.0


def _extract_json(response) -> dict[str, Any]:
    try:
        return response.json() if response.text else {}
    except json.JSONDecodeError:
        return {}


class NeuroSynthUser(HttpUser):
    wait_time = between(0.25, 1.2)

    def on_start(self) -> None:
        # Login first so cookie-based auth is present for all subsequent workflow calls.
        with self.client.post(
            "/auth/login",
            json={
                "username": f"clinician_{random.randint(1, 5000)}",
                "password": "locust-password",
                "role": "CLINICIAN",
            },
            name="POST /auth/login",
            catch_response=True,
        ) as response:
            if response.status_code >= 400:
                response.failure(f"login failed: {response.status_code}")
            else:
                response.success()

    @task
    def clinical_workflow(self) -> None:
        # 1) Clinician opens patient list.
        with self.client.get("/patients", name="GET /patients", catch_response=True) as patients_res:
            if patients_res.status_code >= 400:
                patients_res.failure(f"patients list failed: {patients_res.status_code}")
                return
            patients_payload = _extract_json(patients_res)
            items = patients_payload.get("items", [])

        patient_id = (
            items[random.randrange(len(items))]["patient_id"]
            if items
            else f"P-{random.randint(100, 999)}"
        )

        # 2) Queue prediction for selected patient (path-based API as requested).
        with self.client.post(
            f"/predictions/{patient_id}",
            json={
                "patient_id": patient_id,
                "features": {
                    "Age": 72,
                    "MMSE": 24,
                    "FunctionalAssessment": 6.1,
                    "ADL": 6.0,
                    "SleepQuality": 5.0,
                },
            },
            name="POST /predictions/{id}",
            catch_response=True,
        ) as pred_res:
            if pred_res.status_code >= 400:
                pred_res.failure(f"prediction failed: {pred_res.status_code}")
            else:
                pred_res.success()

        # 3) Retrieve report for same patient.
        with self.client.get(
            f"/reports/{patient_id}",
            name="GET /reports/{id}",
            catch_response=True,
        ) as report_res:
            if report_res.status_code >= 400:
                report_res.failure(f"report failed: {report_res.status_code}")
            else:
                report_res.success()

        # 4) Hold SSE stream for 30 seconds to simulate live biomarker monitoring.
        start = time.time()
        with self.client.get(
            "/biomarkers/stream",
            name="GET /biomarkers SSE",
            stream=True,
            catch_response=True,
        ) as sse_res:
            if sse_res.status_code >= 400:
                sse_res.failure(f"sse failed: {sse_res.status_code}")
                return

            try:
                for line in sse_res.iter_lines(decode_unicode=True):
                    # Keeping the parser very light to avoid adding client-side bottlenecks.
                    _ = line
                    if (time.time() - start) >= 30:
                        break
                sse_res.success()
            except Exception as exc:
                sse_res.failure(f"sse stream error: {exc}")


@events.quitting.add_listener
def fail_on_high_p95(environment, **kwargs) -> None:
    # Locust stores response-time percentiles in milliseconds.
    p95_ms = environment.stats.total.get_response_time_percentile(0.95)
    p95_seconds = p95_ms / 1000.0

    # Human-readable line for CI logs.
    print(f"[Locust Gate] p95 latency: {p95_seconds:.3f}s (threshold: {P95_FAIL_THRESHOLD_SECONDS:.3f}s)")

    if p95_seconds > P95_FAIL_THRESHOLD_SECONDS:
        print("[Locust Gate] FAIL: p95 latency exceeded CI gate")
        environment.process_exit_code = 1
    else:
        print("[Locust Gate] PASS: p95 latency within gate")
        environment.process_exit_code = 0
