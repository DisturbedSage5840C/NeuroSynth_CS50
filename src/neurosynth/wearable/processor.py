# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import asyncpg
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch
from tenacity import retry, stop_after_attempt, wait_exponential

from neurosynth.core.config import NeuroSynthSettings
from neurosynth.core.exceptions import WearableProcessingError
from neurosynth.core.logging import get_logger

BIOMARKER_STREAM_SQL = """
CREATE TABLE IF NOT EXISTS biomarker_stream (
    patient_id UUID NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL,
    biomarker_name TEXT NOT NULL,
    biomarker_value DOUBLE PRECISION,
    source_system TEXT
);
SELECT create_hypertable('biomarker_stream', 'recorded_at', if_not_exists => TRUE, chunk_time_interval => INTERVAL '7 days');
ALTER TABLE biomarker_stream SET (
  timescaledb.compress,
  timescaledb.compress_orderby = 'recorded_at DESC',
  timescaledb.compress_segmentby = 'patient_id'
);
SELECT add_compression_policy('biomarker_stream', INTERVAL '30 days', if_not_exists => TRUE);
"""

WEARABLE_RAW_SQL = """
CREATE TABLE IF NOT EXISTS wearable_raw (
    patient_id UUID NOT NULL,
    device_id TEXT NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL,
    x_g REAL,
    y_g REAL,
    z_g REAL,
    temp_c REAL,
    light_lux REAL
);
SELECT create_hypertable('wearable_raw', 'recorded_at', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');
ALTER TABLE wearable_raw SET (
  timescaledb.compress,
  timescaledb.compress_orderby = 'recorded_at DESC',
  timescaledb.compress_segmentby = 'patient_id'
);
SELECT add_compression_policy('wearable_raw', INTERVAL '7 days', if_not_exists => TRUE);
"""

COMPUTED_FEATURES_SQL = """
CREATE TABLE IF NOT EXISTS computed_features (
    patient_id UUID NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    window_duration_secs INT NOT NULL,
    gait_cadence DOUBLE PRECISION,
    tremor_index DOUBLE PRECISION,
    bradykinesia_score DOUBLE PRECISION,
    freezing_episodes INT,
    step_count INT,
    sleep_duration_mins INT
);
SELECT create_hypertable('computed_features', 'window_start', if_not_exists => TRUE);
"""

DAILY_AGG_SQL = """
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_motor_summary
WITH (timescaledb.continuous) AS
SELECT
  patient_id,
  time_bucket('1 day', window_start) AS date,
  avg(gait_cadence) AS avg_gait_cadence,
  stddev(gait_cadence) AS std_gait_cadence,
  avg(tremor_index) AS avg_tremor_index,
  stddev(tremor_index) AS std_tremor_index,
  avg(bradykinesia_score) AS avg_bradykinesia,
  stddev(bradykinesia_score) AS std_bradykinesia
FROM computed_features
GROUP BY patient_id, date;
"""

WEEKLY_DETERIORATION_SQL = """
CREATE MATERIALIZED VIEW IF NOT EXISTS weekly_deterioration_index
WITH (timescaledb.continuous) AS
SELECT
  patient_id,
  time_bucket('1 week', window_start) AS week,
  0.4 * (avg(tremor_index) - avg(gait_cadence))
  + 0.4 * avg(bradykinesia_score)
  + 0.2 * avg(freezing_episodes) AS deterioration_index
FROM computed_features
GROUP BY patient_id, week;
"""


class WearableStreamProcessor:
    def __init__(self, settings: NeuroSynthSettings) -> None:
        self._settings = settings
        self._logger = get_logger(__name__)
        self._pool: asyncpg.Pool | None = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(dsn=self._settings.timescale_dsn, min_size=1, max_size=8)

    async def initialize_schema(self) -> None:
        if not self._pool:
            raise WearableProcessingError("TimescaleDB pool not connected")
        async with self._pool.acquire() as conn:
            for statement in [BIOMARKER_STREAM_SQL, WEARABLE_RAW_SQL, COMPUTED_FEATURES_SQL, DAILY_AGG_SQL, WEEKLY_DETERIORATION_SQL]:
                await conn.execute(statement)

    async def ingest_stream(self, device_queue: asyncio.Queue) -> None:
        if not self._pool:
            raise WearableProcessingError("TimescaleDB pool not connected")

        batch: list[tuple[Any, ...]] = []
        while True:
            item = await device_queue.get()
            if item is None:
                break

            sample = item["sample"]
            ts = datetime.fromtimestamp(sample["timestamp_ns"] / 1e9, tz=timezone.utc)
            batch.append(
                (
                    UUID(sample["patient_id"]),
                    sample["device_id"],
                    ts,
                    sample["x_g"],
                    sample["y_g"],
                    sample["z_g"],
                    sample["temperature_c"],
                    sample["light_lux"],
                )
            )

            if len(batch) >= 1000:
                await self._copy_rows(batch)
                batch.clear()

        if batch:
            await self._copy_rows(batch)

    async def _copy_rows(self, rows: Sequence[tuple[Any, ...]]) -> None:
        if not self._pool:
            raise WearableProcessingError("TimescaleDB pool not connected")
        async with self._pool.acquire() as conn:
            await conn.copy_records_to_table(
                "wearable_raw",
                records=rows,
                columns=["patient_id", "device_id", "recorded_at", "x_g", "y_g", "z_g", "temp_c", "light_lux"],
            )

    async def compute_gait_features(self, patient_id: UUID, window_hours: int = 24) -> dict[str, float]:
        if not self._pool:
            raise WearableProcessingError("TimescaleDB pool not connected")

        since = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT recorded_at, x_g, y_g, z_g
                FROM wearable_raw
                WHERE patient_id = $1 AND recorded_at >= $2
                ORDER BY recorded_at
                """,
                patient_id,
                since,
            )

            if not rows:
                return {"gait_cadence": 0.0, "tremor_index": 0.0, "bradykinesia_score": 0.0}

            arr = np.array([[r["x_g"], r["y_g"], r["z_g"]] for r in rows], dtype=np.float64)
            mag = np.linalg.norm(arr, axis=1)
            fs = 50.0

            freqs, pxx = welch(mag, fs=fs, nperseg=min(1024, len(mag)))
            gait_mask = (freqs >= 0.5) & (freqs <= 3.0)
            tremor_mask = (freqs >= 4.0) & (freqs <= 8.0)

            gait_cadence = float(freqs[gait_mask][np.argmax(pxx[gait_mask])] * 60.0) if gait_mask.any() else 0.0
            tremor_index = float(np.trapz(pxx[tremor_mask], freqs[tremor_mask])) if tremor_mask.any() else 0.0

            hist, _ = np.histogram(mag, bins=32, density=True)
            hist = hist[hist > 0]
            bradykinesia_score = float(-np.sum(hist * np.log(hist)))
            peaks, _ = find_peaks(mag, distance=10)
            step_count = int(len(peaks))

            features = {
                "gait_cadence": gait_cadence,
                "tremor_index": tremor_index,
                "bradykinesia_score": bradykinesia_score,
                "freezing_episodes": int((mag < np.percentile(mag, 10)).sum() // 100),
                "step_count": step_count,
                "sleep_duration_mins": int((mag < 0.05).sum() / fs / 60),
            }

            await conn.execute(
                """
                INSERT INTO computed_features (
                    patient_id, window_start, window_duration_secs, gait_cadence, tremor_index,
                    bradykinesia_score, freezing_episodes, step_count, sleep_duration_mins
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                """,
                patient_id,
                since,
                window_hours * 3600,
                features["gait_cadence"],
                features["tremor_index"],
                features["bradykinesia_score"],
                features["freezing_episodes"],
                features["step_count"],
                features["sleep_duration_mins"],
            )
            return features

    async def get_deterioration_trajectory(self, patient_id: UUID) -> pd.DataFrame:
        if not self._pool:
            raise WearableProcessingError("TimescaleDB pool not connected")

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT week, deterioration_index
                FROM weekly_deterioration_index
                WHERE patient_id = $1
                ORDER BY week
                """,
                patient_id,
            )

        frame = pd.DataFrame([dict(r) for r in rows])
        if frame.empty:
            return frame
        frame["first_diff"] = frame["deterioration_index"].diff()
        frame["second_diff"] = frame["first_diff"].diff()
        threshold = frame["second_diff"].mean() + 2 * frame["second_diff"].std(ddof=0)
        frame["acceleration_flag"] = frame["second_diff"] > threshold
        return frame
