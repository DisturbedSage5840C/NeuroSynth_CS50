# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

import asyncio
import struct
from collections import deque
from typing import Any
from uuid import UUID

import numpy as np
from asyncio_mqtt import Client, MqttError
from scipy.signal import welch

from neurosynth.connectors.base import AbstractNeuroDataSource
from neurosynth.core.config import NeuroSynthSettings
from neurosynth.core.exceptions import DataIngestionError
from neurosynth.core.logging import get_logger
from neurosynth.core.models import WearableRawSample


class WearableStreamConnector(AbstractNeuroDataSource):
    def __init__(self, settings: NeuroSynthSettings) -> None:
        self._settings = settings
        self._logger = get_logger(__name__)
        self._client: Client | None = None
        self._buffer: deque[WearableRawSample] = deque(maxlen=4096)

    async def connect(self) -> None:
        self._client = Client(hostname=self._settings.mqtt_host, port=self._settings.mqtt_port)

    async def validate_schema(self) -> None:
        if self._client is None:
            raise DataIngestionError("Wearable connector is not connected")

    def _parse_bin_packet(self, payload: bytes, patient_id: UUID, device_id: str) -> WearableRawSample:
        # Binary layout: qfffff = timestamp_ns, x, y, z, temperature, light
        timestamp_ns, x_g, y_g, z_g, temp_c, light_lux = struct.unpack("<qfffff", payload)
        return WearableRawSample(
            patient_id=patient_id,
            device_id=device_id,
            timestamp_ns=timestamp_ns,
            x_g=x_g,
            y_g=y_g,
            z_g=z_g,
            temperature_c=temp_c,
            light_lux=light_lux,
        )

    def _compute_features(self) -> dict[str, float]:
        if len(self._buffer) < 128:
            return {"gait_cadence": 0.0, "tremor_index": 0.0, "bradykinesia_score": 0.0}

        arr = np.array([[s.x_g, s.y_g, s.z_g] for s in self._buffer], dtype=np.float64)
        mag = np.linalg.norm(arr, axis=1)

        fs = 50.0
        freqs, power = welch(mag, fs=fs, nperseg=min(256, len(mag)))

        gait_band = (freqs >= 0.5) & (freqs <= 3.0)
        tremor_band = (freqs >= 4.0) & (freqs <= 8.0)
        gait_cadence = float(freqs[gait_band][np.argmax(power[gait_band])] * 60.0) if gait_band.any() else 0.0
        tremor_index = float(np.trapz(power[tremor_band], freqs[tremor_band])) if tremor_band.any() else 0.0

        hist, _ = np.histogram(mag, bins=32, density=True)
        hist = hist[hist > 0]
        bradykinesia_score = float(-np.sum(hist * np.log(hist)))

        return {
            "gait_cadence": gait_cadence,
            "tremor_index": tremor_index,
            "bradykinesia_score": bradykinesia_score,
        }

    async def fetch_batch(self, offset: int, limit: int) -> list[dict[str, Any]]:
        samples = list(self._buffer)[offset : offset + limit]
        return [sample.model_dump() for sample in samples]

    async def stream(self, queue: asyncio.Queue) -> None:
        if not self._client:
            raise DataIngestionError("Wearable connector is not connected")

        try:
            async with self._client as client:
                async with client.filtered_messages("wearables/+/+/raw") as messages:
                    await client.subscribe("wearables/+/+/raw")
                    async for message in messages:
                        _, patient_s, device_id, _ = message.topic.split("/")
                        sample = self._parse_bin_packet(message.payload, UUID(patient_s), device_id)
                        self._buffer.append(sample)
                        features = self._compute_features()
                        await queue.put({"sample": sample.model_dump(), "features": features})
        except MqttError as exc:
            self._logger.error("wearable.mqtt_error", error=str(exc))
            raise DataIngestionError(str(exc)) from exc
