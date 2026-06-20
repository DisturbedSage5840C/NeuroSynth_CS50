from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

class WearableSimulator:
    def __init__(self, seed: int | None = None, phi: float = 0.85) -> None:
        self.rng = np.random.default_rng(seed)
        self.phi = phi
        self.state = {
            "heartRate": 76.0,
            "spo2": 97.5,
            "systolicBP": 126.0,
            "diastolicBP": 79.0,
            "respiratoryRate": 15.0,
        }

    def _step(self, key: str, mean: float, sigma: float, low: float, high: float) -> float:
        prev = self.state[key]
        innovation = self.rng.normal(0.0, sigma)
        value = mean + self.phi * (prev - mean) + innovation
        value = float(np.clip(value, low, high))
        self.state[key] = value
        return value

    def next_reading(self, patient_id: str) -> dict[str, object]:
        heart_rate = round(self._step("heartRate", mean=76.0, sigma=2.2, low=52.0, high=128.0), 1)
        spo2 = round(self._step("spo2", mean=97.5, sigma=0.25, low=90.0, high=100.0), 1)
        systolic = round(self._step("systolicBP", mean=126.0, sigma=2.8, low=90.0, high=180.0), 1)
        diastolic = round(self._step("diastolicBP", mean=79.0, sigma=2.0, low=55.0, high=115.0), 1)
        respiratory = round(self._step("respiratoryRate", mean=15.0, sigma=0.8, low=9.0, high=28.0), 1)

        return {
            "patient_id": patient_id,
            "time": datetime.now(tz=UTC).strftime("%H:%M:%S"),
            "heartRate": heart_rate,
            "spo2": spo2,
            "systolicBP": systolic,
            "diastolicBP": diastolic,
            "respiratoryRate": respiratory,
        }
