# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""NeuroSynth v5 load test.

Usage:
    pip install locust
    locust -f scripts/load_test.py --headless -u 50 -r 5 --run-time 60s

Target: p95 latency ≤ 2s at 50 concurrent users (plan §5.9 success metric).

User classes:
    NeuroSynthUser       — realistic clinical session (mixed v1/v2/v3, wait 1-3s)
    NeuroSynthV3User     — v3-only load (cross-attention fusion + data endpoints)
    NeuroSynthStressUser — high-throughput v2 burst (wait 0.1-0.5s)

Set NEUROSYNTH_API_URL to point at staging/prod before running.
"""
from __future__ import annotations

import os
import random

from locust import HttpUser, between, task

BASE_URL = os.getenv("NEUROSYNTH_API_URL", "http://localhost:8000")

# ── Feature definitions ───────────────────────────────────────────────────────

FEATURES = [
    "Age", "Gender", "Ethnicity", "EducationLevel", "BMI", "Smoking",
    "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality",
    "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes",
    "Depression", "HeadInjury", "Hypertension", "SystolicBP", "DiastolicBP",
    "CholesterolTotal", "CholesterolLDL", "CholesterolHDL",
    "CholesterolTriglycerides", "MMSE", "FunctionalAssessment", "MemoryComplaints",
    "BehavioralProblems", "ADL", "Confusion", "Disorientation",
    "PersonalityChanges", "DifficultyCompletingTasks", "Forgetfulness",
    # v5 extensions
    "UPDRS_motor", "UPDRS_total", "tremor_amplitude", "gait_velocity",
    "APOE4_dosage", "CSF_Abeta42", "CSF_pTau", "nWBV", "eTIV",
]

RANGES: dict[str, tuple[float, float]] = {
    "Age": (50, 90), "Gender": (0, 1), "Ethnicity": (0, 3),
    "EducationLevel": (5, 20), "BMI": (18, 40), "Smoking": (0, 1),
    "AlcoholConsumption": (0, 20), "PhysicalActivity": (0, 10),
    "DietQuality": (0, 10), "SleepQuality": (2, 10),
    "FamilyHistoryAlzheimers": (0, 1), "CardiovascularDisease": (0, 1),
    "Diabetes": (0, 1), "Depression": (0, 1), "HeadInjury": (0, 1),
    "Hypertension": (0, 1), "SystolicBP": (90, 180),
    "DiastolicBP": (60, 110), "CholesterolTotal": (150, 300),
    "CholesterolLDL": (50, 200), "CholesterolHDL": (30, 80),
    "CholesterolTriglycerides": (50, 400), "MMSE": (10, 30),
    "FunctionalAssessment": (1, 10), "MemoryComplaints": (0, 1),
    "BehavioralProblems": (0, 1), "ADL": (1, 10), "Confusion": (0, 1),
    "Disorientation": (0, 1), "PersonalityChanges": (0, 1),
    "DifficultyCompletingTasks": (0, 1), "Forgetfulness": (0, 1),
    "UPDRS_motor": (0, 108), "UPDRS_total": (0, 176),
    "tremor_amplitude": (0, 5), "gait_velocity": (0.2, 1.8),
    "APOE4_dosage": (0, 2), "CSF_Abeta42": (200, 1700),
    "CSF_pTau": (15, 120), "nWBV": (0.6, 0.9), "eTIV": (1100, 1800),
}

LITERATURE_QUERIES = [
    "MMSE cognitive decline Alzheimer",
    "dopamine deficiency Parkinson's disease treatment",
    "APOE4 amyloid beta plaques",
    "ALS motor neuron disease SOD1",
    "multiple sclerosis white matter lesions MRI",
    "Huntington CAG repeat neurodegeneration",
]


def random_patient() -> dict:
    patient: dict[str, float] = {}
    for f in FEATURES:
        lo, hi = RANGES.get(f, (0, 1))
        patient[f] = random.choice([0, 1]) if hi <= 1 else round(random.uniform(lo, hi), 2)
    return patient


# ── Realistic clinical user (v1 + v2 + v3 mixed) ─────────────────────────────

class NeuroSynthUser(HttpUser):
    """Simulates a clinician: runs analysis, reads reports, searches literature."""

    host = BASE_URL
    wait_time = between(1, 3)

    def on_start(self) -> None:
        resp = self.client.post("/auth/login", json={
            "username": "loadtest", "password": "loadtest123",
        })
        if resp.status_code == 200:
            token = resp.json().get("access_token", "")
            self.client.headers["Authorization"] = f"Bearer {token}"

    @task(4)
    def v3_analyze(self) -> None:
        """v3 analysis with cross-attention fusion (primary path)."""
        self.client.post(
            "/v3/predictions/analyze",
            json={"patient_id": f"LT-{random.randint(1000, 9999)}", "features": random_patient()},
            name="/v3/predictions/analyze",
        )

    @task(3)
    def v2_analyze(self) -> None:
        """v2 analysis — legacy path, lower weight."""
        self.client.post(
            "/v2/predictions/analyze",
            json={"patient_id": f"LT-{random.randint(1000, 9999)}", "features": random_patient()},
            name="/v2/predictions/analyze",
        )

    @task(2)
    def cohort_stats(self) -> None:
        """Population stats — cached, should be < 50ms."""
        self.client.get("/v3/data/cohort/stats", name="/v3/data/cohort/stats")

    @task(2)
    def data_sources(self) -> None:
        """Data source list — lightweight DB read."""
        self.client.get("/v3/data/sources", name="/v3/data/sources")

    @task(1)
    def literature_search(self) -> None:
        """pgvector similarity search — moderate latency."""
        self.client.post(
            "/v3/literature/search",
            json={"query": random.choice(LITERATURE_QUERIES), "top_k": 5},
            name="/v3/literature/search",
        )

    @task(1)
    def fusion_weights(self) -> None:
        """Modality fusion weights — very fast static read."""
        self.client.get("/v3/fusion/weights", name="/v3/fusion/weights")

    @task(1)
    def provenance(self) -> None:
        """Data provenance lineage."""
        self.client.get("/v3/data/provenance", name="/v3/data/provenance")

    @task(3)
    def ready_check(self) -> None:
        """Readiness probe — should always be < 30ms."""
        self.client.get("/ready", name="/ready")

    @task(1)
    def v2_report(self) -> None:
        """SOAP report generation — least frequent, highest latency."""
        self.client.post(
            "/v2/reports/generate",
            json={"patient_id": f"LT-{random.randint(1000, 9999)}"},
            name="/v2/reports/generate",
        )


# ── v3-only user (validates new endpoints under sustained load) ───────────────

class NeuroSynthV3User(HttpUser):
    """Exclusively hits v3 endpoints to validate new backend under load."""

    host = BASE_URL
    wait_time = between(0.5, 2)

    def on_start(self) -> None:
        resp = self.client.post("/auth/login", json={
            "username": "loadtest", "password": "loadtest123",
        })
        if resp.status_code == 200:
            self.client.headers["Authorization"] = f"Bearer {resp.json().get('access_token', '')}"

    @task(5)
    def v3_analyze(self) -> None:
        self.client.post(
            "/v3/predictions/analyze",
            json={"patient_id": f"V3-{random.randint(1, 9999)}", "features": random_patient()},
            name="/v3/predictions/analyze [v3-user]",
        )

    @task(2)
    def literature_search(self) -> None:
        self.client.post(
            "/v3/literature/search",
            json={"query": random.choice(LITERATURE_QUERIES), "top_k": 3},
            name="/v3/literature/search [v3-user]",
        )

    @task(2)
    def cohort_stats(self) -> None:
        self.client.get("/v3/data/cohort/stats", name="/v3/data/cohort/stats [v3-user]")

    @task(1)
    def literature_status(self) -> None:
        self.client.get("/v3/literature/status", name="/v3/literature/status")


# ── Stress burst user ─────────────────────────────────────────────────────────

class NeuroSynthStressUser(HttpUser):
    """High-throughput burst: validates p95 ≤ 2s under 50-user concurrency."""

    host = BASE_URL
    wait_time = between(0.1, 0.5)

    def on_start(self) -> None:
        resp = self.client.post("/auth/login", json={
            "username": "loadtest", "password": "loadtest123",
        })
        if resp.status_code == 200:
            self.client.headers["Authorization"] = f"Bearer {resp.json().get('access_token', '')}"

    @task(3)
    def rapid_v3_analyze(self) -> None:
        self.client.post(
            "/v3/predictions/analyze",
            json={"patient_id": f"STRESS-{random.randint(1, 99999)}", "features": random_patient()},
            name="/v3/predictions/analyze [stress]",
        )

    @task(2)
    def rapid_v2_analyze(self) -> None:
        self.client.post(
            "/v2/predictions/analyze",
            json={"patient_id": f"STRESS-{random.randint(1, 99999)}", "features": random_patient()},
            name="/v2/predictions/analyze [stress]",
        )

    @task(1)
    def rapid_cohort(self) -> None:
        self.client.get("/v3/data/cohort/stats", name="/v3/data/cohort/stats [stress]")
